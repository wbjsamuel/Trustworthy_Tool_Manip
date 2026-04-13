from policy.pose_dp2_dino import DP2PoseDINO


class PoseConditionedDINO_DP2(DP2PoseDINO):
    """
    Stage-1-conditioned DP2-DINO policy.

    This is a thin alias over the existing Stage 1 + DP2 integration so it can
    be referenced by a clearer policy/config name for training and inference.
    """

    pass
