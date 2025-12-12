import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from GDNSM.utils.quick_start import quick_start
import torch.utils.tensorboard as tb
os.environ['NUMEXPR_MAX_THREADS'] = '48'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GDNSM', help='name of models') 
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--mg', action="store_true", help='whether to use Mirror Gradient, default is False')

    # ==========================================
    # [新增] UFNRec 核心超参数
    # ==========================================
    parser.add_argument('--reverse', type=int, default=2, help='UFNRec: 判定 FN 的阈值 m (连续多少次高分)')
    parser.add_argument('--lbd', type=float, default=0.1, help='UFNRec: Consistency Loss 的权重 alpha')
    parser.add_argument('--decay', type=float, default=0.999, help='UFNRec: EMA Teacher 的衰减率')
    parser.add_argument('--ufn_warmup', type=int, default=5, help='UFNRec: 热身 Epoch 数，在此之前不开启挖掘')
    # ==========================================

    # 解析参数，这是为了下面 config_dict 能用到 args
    args, _ = parser.parse_known_args()


    config_dict = {
    
    # ==========================================
    # [新增] 注入 UFNRec 参数到 Config
    # ==========================================
    'reverse': args.reverse,
    'lbd': args.lbd,
    'decay': args.decay,
    'ufn_warmup': args.ufn_warmup,
    # ==========================================
    
    # 基本训练参数
    'learning_rate_scheduler': [0.96, 50],
    'stopping_step': 50,
    'clip_grad_norm': {'max_norm': 5.0},
    'req_training': True,
    'eval_type': 'full',
    'learner': 'adam',
    'eval_step': 1,
    'loss_type': 'l2',  # 论文实验 L2 loss 最优

    # 模型相关
    'cl_loss': 1.0,
    'knn_k': 20,
    'embedding_size': 64,
    'n_ui_layers': 2,          # User-Item GCN 层数
    'n_layers': 2,             # Item-Item GCN 层数
    'reg_weight': 1e-4,        # BPR正则权重

    # diffusion 模型参数
    'use_mm_diff' : True,
    'd_epoch' : 5,
    'timesteps': 5,           # diffusion 步数 T
    'beta':1.0,
    'diffusion_T': 50,
    'cfg_scale_text': 1.1,     # MG 数值，文本
    'cfg_scale_visual': 1.1,   # MG 数值，图像
    'mg': True,                # 开启 modality guidance
    'ds': True,                # 开启 dynamic difficulty scheduling

    'num_generated_neg': 6,    # 每个正样本生成负样本数量
    'smoothing_S': 30,         # 平滑系数
    'lambda_ds': 1,            # DS 系数

    # 数据集相关
    'dataset': 'baby',
    'data_path': '/home/ma-user/work/GDNSM_UFNRec/root/shared-nvme/GDNSM-main/GDNSM/dataset/',
    'inter_file_name': 'baby.inter',
    'USER_ID_FIELD': 'userID',
    'ITEM_ID_FIELD': 'itemID',
    'inter_splitting_label': 'x_label',
    'field_separator': '\t',
    'filter_out_cod_start_users': True,
    'NEG_PREFIX': 'neg_',
    'text_feat_file': 'text_feat.npy',
    'image_feat_file': 'image_feat.npy',

    # 训练/评估
    'cfg_scale_text':1.0,
    'train_batch_size': 2048,
    'eval_batch_size': 2048,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'epochs': 300,
    'valid_metric': 'Recall@10',
    'metrics': ['Recall', 'NDCG'],
    'topk': [10, 20],
    'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'RO', 'mode': 'full'},
    'valid_metric_bigger': True,

    # GPU
    'use_gpu': True,
    'gpu_id': 0,

    # 随机种子
    'hyper_parameters': ['seed'],
    'seed': [2025],

    # 内部调试或兼容
    'model': 'GDNSM',
    'device': 'cuda'
}


    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True, mg=args.mg)


