from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """训练专用参数类，包含 BaseOptions 的共享参数"""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        # 新增消融实验参数
        parser.add_argument('--lambda_sym', type=float, default=5.0, help='weight for symmetry loss')
        parser.add_argument('--lambda_struct', type=float, default=10.0, help='weight for structure loss')

        # HTML 可视化参数
        parser.add_argument('--display_freq', type=int, default=400, help='显示训练结果频率')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='保存训练结果到HTML频率')
        parser.add_argument('--print_freq', type=int, default=100, help='控制台显示训练信息频率')
        parser.add_argument('--no_html', action='store_true', help='不保存中间训练结果')

        # 模型保存/加载参数
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='保存最新结果频率')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='每几个 epoch 保存模型')
        parser.add_argument('--save_by_iter', action='store_true', help='是否按迭代保存模型')
        parser.add_argument('--continue_train', action='store_true', help='继续训练：加载最新模型')
        parser.add_argument('--epoch_count', type=int, default=1, help='起始 epoch')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test 等')

        # 训练参数
        parser.add_argument('--n_epochs', type=int, default=100, help='初始学习率训练 epoch 数')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='学习率线性衰减 epoch 数')
        parser.add_argument('--beta1', type=float, default=0.5, help='Adam 动量系数')
        parser.add_argument('--lr', type=float, default=0.0002, help='初始学习率')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='GAN loss 类型')
        parser.add_argument('--pool_size', type=int, default=50, help='图像缓存池大小')
        parser.add_argument('--lr_policy', type=str, default='linear', help='学习率策略')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='每多少次迭代乘 gamma')

        self.isTrain = True
        return parser


class TestOptions(BaseOptions):
    """This class includes test options."""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.set_defaults(model='cycle_gan')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time')
        parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        # ⚠️ 新增 symmetry loss 参数，让测试也可以解析
        parser.add_argument('--lambda_sym', type=float, default=10.0, help='weight for symmetry loss Lsym (optional for test)')

        self.isTrain = False
        return parser