
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "value"])
AgentInfoRnn = namedarraytuple("AgentInfoRnn", ["dist_info", "value", "prev_rnn_state"])
NdigoInfo = namedarraytuple("NdigoInfo", ["prev_gru_state"])
IcmInfo = namedarraytuple("IcmInfo", [])