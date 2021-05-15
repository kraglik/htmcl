struct cell;
struct column;
struct synapse;
struct proximal_synapse;
struct dendrite;
struct apical_dendrite;
struct layer;
struct layer_connection;
struct htm_config;
struct htm;
struct sdr;

bool cell_was_active(global struct cell* c);
global struct cell* get_cell_from_column(global struct column* col, unsigned int cell_pos);
global struct cell* get_cell_from_layer(global struct layer* l, unsigned int cell_id);
