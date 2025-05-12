from .run_metric import RunMetric
from .run_no_obj_metric import RunNoObjectMetric
from .layup_metric import LayupMetric
from .shot_metric import ShotMetric
# from .chestpass_metric import ChestpassMetric
from .jumpshot_metric import JumpshotMetric
from .setshot_metric import SetshotMetric
from .getup_metric import GetupMetric
from .place_metric import PlaceMetric
from .drink_metric import DrinkMetric
from .pour_metric import PourMetric
from .chair_metric import ChairMetric
from .pickup_metric import PickupMetric
from .multipour_metric import MultiPourMetric

METRIC_ARGS = {
    'run': [],
    'rrun': [],
    'lrun': [],
    'layup': ['layup_target'],
    'shot': ['layup_target'],
    'Chestpass': ['layup_target'],
    'Jumpshot': ['layup_target'],
    'Setshot': ['layup_target'],
    'turnlayup': ['layup_target'],
    'run_no_object':['switch_skill_name'],
    'getup': [],
    'place_pan': [],
    'place_book': [],
    'drink_cup': [],
}

# skillmimic/metrics/metric_factory.py
def create_metric(skill_name, num_envs, device, **kwargs):
    metric_classes = {
        'run': RunMetric,
        'rrun': RunMetric,
        'lrun': RunMetric,
        'layup': LayupMetric,
        'shot': ShotMetric,
        'turnlayup': LayupMetric,
        'Chestpass': LayupMetric, #ChestpassMetric,
        'pickup': PickupMetric,
        'Jumpshot': JumpshotMetric,
        'Setshot': SetshotMetric,
        'run_no_object':RunNoObjectMetric,
        'getup': GetupMetric,
        'place_pan': PlaceMetric,
        'place_kettle': PlaceMetric,
        'place_book': PlaceMetric,
        'drink_cup': DrinkMetric,
        'pour_kettle': PourMetric,
        'getup_chair': ChairMetric,
        'pour_kettlecup': MultiPourMetric,
        # 添加其他技能和对应的 Metric 类
    }

    MetricClass = metric_classes.get(skill_name)
    allowed_keys = METRIC_ARGS.get(skill_name, [])
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
    if MetricClass:
        return MetricClass(num_envs, device, **filtered_kwargs)
    else:
        return None  # 或者返回一个默认的 Metric
