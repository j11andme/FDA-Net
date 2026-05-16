from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
import numpy as np               
import torch.distributed as dist 
from ultralytics.utils import LOGGER, colorstr, RANK, TQDM
from ultralytics.utils.torch_utils import unwrap_model, autocast, unset_deterministic
import time
import math
import torch
from torch import nn
from ultralytics.utils.enhence_img import frequency_mixup
from ultralytics.utils.loss import FSA_LOSS
from ultralytics.cfg import DEFAULT_CFG

class DuDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.hook_handles = []
        self.captured_feats = {}  
        self.source_layers = []   
        
        self.consistency_weight = 3.0

    def _hook_fn(self, module, input, output, layer_idx):
        self.captured_feats[layer_idx] = output

    def _register_hooks(self):
        if self.hook_handles: return 
        
        m = unwrap_model(self.model)
        if hasattr(m.model[-1], 'f'):
            self.source_layers = m.model[-1].f
        else:
            self.source_layers = [len(m.model)-4, len(m.model)-3, len(m.model)-2]

        LOGGER.info(f"Dual-Stream: Hooking layers {self.source_layers} for FSA Loss")
        
        for i in self.source_layers:
            layer = m.model[i]
            handle = layer.register_forward_hook(
                lambda m, i, o, idx=i: self._hook_fn(m, i, o, idx)
            )
            self.hook_handles.append(handle)

    def _do_train(self):
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        
        self._register_hooks()

        epoch = self.start_epoch
        self.optimizer.zero_grad()
        
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            self._model_train()
            
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            
            pbar = enumerate(self.train_loader)
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            
            self.tloss = None
            
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)])
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    
                    self.captured_feats = {} 
                    
                    if self.args.compile:
                        preds = self.model(batch["img"])
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        preds = self.model(batch["img"]) 
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    
                    feats_orig = [self.captured_feats[idx] for idx in self.source_layers]
                    
                    loss_fsa = torch.tensor(0.0, device=self.device)
                    loss_aug_det = torch.tensor(0.0, device=self.device)
                    
                    self.captured_feats = {} 
                    
                    #  LFM 
                    with torch.no_grad():
                        img_aug = frequency_mixup(batch["img"])
                    
                    #  Forward 
                    preds_aug = self.model(img_aug) 
                    
                    feats_aug = [self.captured_feats[idx] for idx in self.source_layers]
                    
                    loss_fsa = FSA_LOSS(feats_orig, feats_aug, batch, batch["img"].shape) * self.consistency_weight
                    
                    loss_aug_base, _ = unwrap_model(self.model).loss(batch, preds_aug)
                    loss_aug_det = loss_aug_base.sum() 
                    
                    self.loss = loss.sum() + loss_fsa + loss_aug_det
                    
                    if RANK != -1: self.loss *= self.world_size
                    self.loss_items = torch.cat([
                        self.loss_items, 
                        loss_fsa.detach().view(1), 
                        loss_aug_det.detach().view(1)
                    ])
                    
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()
                
                # Optimize
                if ni - last_opt_step >= self.accumulate:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.ema:
                        self.ema.update(self.model)
                    last_opt_step = ni

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(threshold=0.5)
                    self.metrics, self.fitness = self.validate()

                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs

            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)

            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1

        if RANK in {-1, 0}:
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Add fsa_loss and aug_loss to log names"""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        keys.append(f"{prefix}/fsa_loss")
        keys.append(f"{prefix}/aug_loss")
        
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys
    
    def get_validator(self):
        val = super().get_validator()
        self.loss_names = ["box_loss", "cls_loss", "dfl_loss", "fsa_loss", "aug_loss"]
        return val
    
