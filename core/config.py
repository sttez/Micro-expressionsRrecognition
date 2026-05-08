import configparser
import os

class Config:
    """单例配置管理器，项目内唯一配置入口"""

    _instance = None

    def __new__(cls, config_path='config.ini'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path='config.ini'):
        if self._initialized:
            return
        self._config = configparser.ConfigParser()
        self._config.read(config_path)
        self._project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._initialized = True

    @property
    def project_root(self):
        return self._project_root

    def path(self, section, key):
        """获取路径配置，自动解析为绝对路径"""
        raw = self._config.get(section, key)
        if os.path.isabs(raw):
            return raw
        return os.path.join(self._project_root, raw)

    def get(self, section, key, fallback=None):
        return self._config.get(section, key, fallback=fallback)

    def getint(self, section, key, fallback=None):
        return self._config.getint(section, key, fallback=fallback)

    def getfloat(self, section, key, fallback=None):
        return self._config.getfloat(section, key, fallback=fallback)

    def getboolean(self, section, key, fallback=None):
        return self._config.getboolean(section, key, fallback=fallback)

    def getlist(self, section, key, fallback=None, sep=','):
        """获取逗号分隔的列表配置"""
        try:
            raw = self._config.get(section, key, fallback=fallback)
            if raw is None:
                return fallback
            return [item.strip() for item in raw.split(sep) if item.strip()]
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

    def get_weights_dir(self):
        return self.path('paths', 'weights_dir')

    def get_label_train(self):
        return os.path.join(self._project_root, self.get('paths', 'label_train', 'cls_train.txt'))

    def get_label_test(self):
        return os.path.join(self._project_root, self.get('paths', 'label_test', 'cls_test.txt'))

    def get_class_mapping(self):
        return os.path.join(self._project_root, self.get('paths', 'class_mapping', 'class_mapping.txt'))

    def get_num_classes(self):
        names = self.get_class_names()
        return len(names) if names else 5

    def get_class_names(self):
        result = self.getlist('preprocessing', 'valid_classes')
        if result:
            return result
        # fallback: 尝试旧版 section 名
        result = self.getlist('CLASSES', 'valid_classes')
        return result if result else ['surprise', 'repression', 'happiness', 'disgust', 'others']

    def get_model_path(self):
        """查找模型路径：优先使用用户选定的模型，否则返回最新的"""
        # 检查用户手动选定的模型
        selected = self.get('model', 'selected_path', '')
        if selected and os.path.exists(selected):
            return selected
        # 自动查找最新的已训练模型权重
        weights_dir = self.get_weights_dir()
        if not os.path.exists(weights_dir):
            return None
        best_path = None
        best_time = 0
        for item in os.listdir(weights_dir):
            model_file = os.path.join(weights_dir, item, 'best_model.pth')
            if os.path.exists(model_file):
                mtime = os.path.getmtime(model_file)
                if mtime > best_time:
                    best_time = mtime
                    best_path = model_file
        return best_path

    def set_selected_model(self, model_path):
        """保存用户选定的模型路径到配置文件"""
        self._config.set('model', 'selected_path', model_path or '')
        with open(os.path.join(self._project_root, 'config.ini'), 'w') as f:
            self._config.write(f)

    def list_models(self):
        """列出所有可用的模型，返回 [(名称, 路径, 准确率), ...]"""
        weights_dir = self.get_weights_dir()
        if not os.path.exists(weights_dir):
            return []
        models = []
        for item in os.listdir(weights_dir):
            model_file = os.path.join(weights_dir, item, 'best_model.pth')
            if os.path.exists(model_file):
                # 从文件夹名提取准确率（格式: MicroExpModel_<时间戳>_<准确率>）
                acc = 'N/A'
                parts = item.split('_')
                if len(parts) >= 3:
                    try:
                        acc_val = float(parts[-1])
                        acc = f"{acc_val:.2f}%"
                    except ValueError:
                        pass
                models.append((item, model_file, acc))
        # 按修改时间倒序
        models.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
        return models

    def get_font_path(self):
        """获取字体路径，自动检测系统字体"""
        font = self.get('desktop', 'font_path', '')
        if font and os.path.exists(font):
            return font
        # Windows 默认中文字体
        win_font = r'C:\Windows\Fonts\msyh.ttc'
        if os.path.exists(win_font):
            return win_font
        return None
