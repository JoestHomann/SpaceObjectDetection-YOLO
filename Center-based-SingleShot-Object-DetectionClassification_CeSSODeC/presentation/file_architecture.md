```mermaid
flowchart LR
  %% -------- Entrypoints --------
  subgraph EntryPoints["Entry points"]
    TM["train_main.py\nCLI (training only)\n-> train.fit(RunConfig)"]
    IM["infer_main.py\nCLI (inference only)\n-> infer.run_inference(...)"]
  end

  %% -------- Core config --------
  subgraph Config["Configuration"]
    CFG["config.py\nRunConfig = {DataConfig, GridConfig, ModelConfig, TrainConfig}"]
  end

  %% -------- Training stack --------
  subgraph Training["Training stack"]
    TR["train.py\nbuild_loaders\ntrain_one_epoch\nvalidate\nfit"]
    DS["dataset.py\nSingleObjectYoloDataset\n__getitem__ -> (x, ij_gt, bbox_gt_norm, cls_gt)"]
    TG["targets.py\nyolo_norm_to_ij\nclamp_ij"]
    MD["model.py\nCenterSingleObjNet\nforward -> (center_pred, box_pred, cls_pred)"]
    LS["losses.py\nCenterSingleObjLoss\nforward -> {Lc,Lb,Lk,L}"]
    CK["ckpt.py\nsave_checkpoint\nload_checkpoint"]
  end

  %% -------- Inference stack --------
  subgraph Inference["Inference stack"]
    INF["infer.py\nload_model_for_inference\npreprocess_image\ndecode_single\nrun_inference"]
  end

  %% -------- Wiring --------
  TM --> CFG --> TR
  TR --> DS
  DS --> TG
  TR --> MD
  TR --> LS
  TR --> CK

  IM --> CFG --> INF
  INF --> CK
  INF --> MD
  INF --> DS
  DS --> TG
