from typing import Type

from myrecall.methods.ewc import EWC_SAC
from myrecall.methods.l2 import L2_SAC
from myrecall.methods.packnet import PackNet_SAC
from myrecall.methods.clonex import ClonEx_SAC
from myrecall.methods.recall import RECALL_SAC
from myrecall.methods.agem import AGEM_SAC
from myrecall.methods.mas import MAS_SAC
from myrecall.methods.vcl import VCL_SAC
from myrecall.sac.sac import SAC
from myrecall.methods.TriRL import TriRL_SAC


def get_sac_class(cl_method: str) -> Type[SAC]:
    if cl_method in ["ft", "pm","mtr"]:
        return SAC
    if cl_method == "recall":
        return RECALL_SAC
    if cl_method == "ewc":
        return EWC_SAC
    if cl_method == "packnet":
        return PackNet_SAC
    if cl_method == "clonex":
        return ClonEx_SAC
    if cl_method == "mas":
        return MAS_SAC
    if cl_method == "vcl":
        return VCL_SAC
    if cl_method == "l2":
        return L2_SAC
    if cl_method == "agem":
        return AGEM_SAC
    if cl_method == "3rl":
        return TriRL_SAC
    assert False, "Bad cl_method!"
