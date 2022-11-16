import dataclasses


@dataclasses.dataclass
class AttentionConf:
    embed_dim: int
    num_heads: int
    attn_drop_prob: float 
    """ Dropout probability for attention weights """
    resid_drop_prob: float 
    """ Dropout probability for head mergin layer"""

    def __post_init__(self) -> None:
        assert (
            not self.embed_dim % self.num_heads
        ), f"Embedding dim must be a multiple of number of heads got {self.num_heads} heads and dim of {self.embed_dim}"


@dataclasses.dataclass
class GPTConf:
    attention: AttentionConf
    num_blocks: int
    vocab_size: int
    max_seq_len: int
    embed_pdrop: float




def get_schema(obj):
    assert dataclasses.is_dataclass(obj), f"{obj.__class__} must be a dataclass"
    schema = {}

    for field in dataclasses.fields(obj):
        if dataclasses.is_dataclass(field.type):
            _schema = get_schema(field.type)
            schema[field.name] = (field.type, _schema)

    return schema

def flatten_dataclass(obj):
    assert dataclasses.is_dataclass(obj), f"{obj.__class__} must be a dataclass"

    data = {}

    for field in dataclasses.fields(obj):
        if dataclasses.is_dataclass(field.type):
            _data = flatten_dataclass(getattr(obj, field.name))
            data[field.name] = _data
        else:
            val = getattr(obj, field.name) 
            if field.init:
                data[field.name] = val
            else:
                # TODO: this is a hack. not planning on fixing this in the near future.
                data[f"_generated field: {field.name}"] = val

    return data

def to_struct(cls, data, schema):
    assembeled_ = {}
    for fname, (_cls, _schema) in schema.items():
        if not _schema:
            assembeled_[fname] = _cls(**data[fname])
        else:
            assembeled_[fname] = to_struct(_cls, data[fname], _schema)

    for k,v in data.items():
        if k  in schema:
            continue
        assembeled_[k] = v

    return cls(**assembeled_) 


def build_from_data(cls, data):
    schema = get_schema(cls)

    return to_struct(cls, data, schema)


def dict_map(kv_func, dict_tree) -> dict:
    res = {}
    for k, v in dict_tree.items():
        if isinstance(v, dict):
            res[k] = dict_map(kv_func, v)
        else:
            new_k, new_v = kv_func(k, v) 
            if new_k is not None:
                res[new_k] = new_v

    return res


