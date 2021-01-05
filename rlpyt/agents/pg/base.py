
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "value"])
AgentInfoRnn = namedarraytuple("AgentInfoRnn", ["dist_info", "value", "prev_rnn_state"])
IcmInfo = namedarraytuple("IcmInfo", [])
NdigoInfo = namedarraytuple("NdigoInfo", ["prev_gru_state"])