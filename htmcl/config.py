from dataclasses import dataclass


@dataclass
class LayerConfig:
    layer_size_x: int = 32
    layer_size_y: int = 32

    cells_per_column: int = 8

    distal_segments_per_cell: int = 10
    apical_segments_per_cell: int = 10

    proximal_synapses_per_segment: int = 40
    distal_synapses_per_segment: int = 20
    apical_synapses_per_segment: int = 20

    max_new_distal_synapses: int = 4
    max_new_apical_synapses: int = 4

    segment_activation_threshold: int = 15
    segment_minimal_threshold: int = 10

    neighbors_radius: int = 4

    proximal_permanence_inc: float = 0.01
    distal_permanence_inc: float = 0.01
    apical_permanence_inc: float = 0.01

    proximal_permanence_dec: float = 0.01
    distal_permanence_dec: float = 0.01
    apical_permanence_dec: float = 0.01

    proximal_permanence_limit: float = 1.0
    distal_permanence_limit: float = 1.0
    apical_permanence_limit: float = 1.0

    proximal_permanence_threshold: float = 0.3
    distal_permanence_threshold: float = 0.3
    apical_permanence_threshold: float = 0.3

    proximal_decay: float = 0.001
    distal_decay: float = 0.001
    apical_decay: float = 0.001

    boost_strength: float = 1.0
    boost_increase: float = 0.01
    boost_decrease: float = 0.01

    overlap_threshold: float = 15.0

    learning: bool = True
    input_layer: bool = False
