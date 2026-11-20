"""Associate against the same-frame 2D target, never the largest person."""
import numpy as np
from .mapping import CWB_INDICES


class TargetAssociation:
    def __init__(self, policy):
        self.policy = policy
        self.previous_center = None

    def reset(self):
        self.previous_center = None

    def choose(self, candidates, target):
        target = np.asarray(target)
        if target.shape != (65, 3):
            raise ValueError('invalid_target_shape')
        policy = self.policy
        ranked = []
        for index, candidate in enumerate(candidates):
            xy, scores = np.asarray(candidate.points2d), np.asarray(candidate.scores)
            if xy.shape != (133, 2) or scores.shape != (133,):
                raise ValueError('invalid_candidate_shape')
            xy, scores = xy[list(CWB_INDICES)][:17], scores[list(CWB_INDICES)][:17]
            valid = (np.isfinite(xy).all(axis=1) & np.isfinite(scores)
                     & (xy >= 0).all(axis=1) & (xy <= 1).all(axis=1)
                     & (scores >= policy['minConfidence']) & (scores <= 1)
                     & np.isfinite(target[:17]).all(axis=1)
                     & (target[:17, :2] >= 0).all(axis=1) & (target[:17, :2] <= 1).all(axis=1)
                     & (target[:17, 2] >= policy['minConfidence']) & (target[:17, 2] <= 1))
            if int(valid.sum()) < policy['minCommonJoints']:
                continue
            distance = float(np.mean(np.linalg.norm(xy[valid] - target[:17, :2][valid], axis=1)))
            center = np.median(xy[valid], axis=0)
            if (distance > policy['maxMeanDistance'] or
                    (self.previous_center is not None and
                     np.linalg.norm(center - self.previous_center) > policy['maxCenterJump'])):
                continue
            ranked.append((distance, index, center))
        ranked.sort(key=lambda entry: entry[0])
        if not ranked:
            return None, 'target_unmatched'
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < policy['ambiguityMargin']:
            return None, 'identity_ambiguous'
        self.previous_center = ranked[0][2]
        return ranked[0][1], None
