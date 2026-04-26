"""Label generation and timing safety utilities."""

from quantauto.labels.builders import (
    make_direction_label,
    make_forward_cross_sectional_rank_label,
    make_forward_rank_label,
    make_forward_return_label,
)
from quantauto.labels.workflows import LabelSpec, build_label, build_label_set, validate_label_set
from quantauto.labels.timing import (
    LabelMeta,
    check_label_feature_overlap,
    get_valid_label_range,
    validate_label_timing,
)

__all__ = [
    "LabelMeta",
    "validate_label_timing",
    "get_valid_label_range",
    "check_label_feature_overlap",
    "make_forward_return_label",
    "make_direction_label",
    "make_forward_rank_label",
    "make_forward_cross_sectional_rank_label",
    "LabelSpec",
    "build_label",
    "build_label_set",
    "validate_label_set",
]
