#!/usr/bin python3
# -*- coding: utf-8 -*-

# here put the import lib
import os
import sys
import copy
import json
from collections import defaultdict

sys.path.insert(0, os.path.abspath(""))
from utils.fetch_features import *


class mask:

    MOTION_MASK = defaultdict(list)

    MOTION_MASK["move"] = [
        "l_wheel_joint_position",
        "r_wheel_joint_position",
        "l_wheel_joint_velocity",
        "r_wheel_joint_velocity",
        "l_wheel_joint_effort",
        "r_wheel_joint_effort",
        "fetch_pose_position_x",
        "fetch_pose_position_y",
        "fetch_pose_position_z",
        "fetch_pose_orientation_x",
        "fetch_pose_orientation_y",
        "fetch_pose_orientation_z",
        "fetch_pose_orientation_w",
        "fetch_twist_linear_x",
        "fetch_twist_linear_y",
        "fetch_twist_linear_z",
        "fetch_twist_angular_x",
        "fetch_twist_angular_y",
        "fetch_twist_angular_z",
    ]

    MOTION_MASK["pick_cube"] = [
        "torso_lift_joint_position",
        "shoulder_pan_joint_position",
        "shoulder_lift_joint_position",
        "upperarm_roll_joint_position",
        "elbow_flex_joint_position",
        "forearm_roll_joint_position",
        "wrist_flex_joint_position",
        "wrist_roll_joint_position",
        "l_gripper_finger_joint_position",
        "r_gripper_finger_joint_position",
        "torso_lift_joint_velocity",
        "shoulder_pan_joint_velocity",
        "shoulder_lift_joint_velocity",
        "upperarm_roll_joint_velocity",
        "elbow_flex_joint_velocity",
        "forearm_roll_joint_velocity",
        "wrist_flex_joint_velocity",
        "wrist_roll_joint_velocity",
        "l_gripper_finger_joint_velocity",
        "r_gripper_finger_joint_velocity",
        "torso_lift_joint_effort",
        "shoulder_pan_joint_effort",
        "shoulder_lift_joint_effort",
        "upperarm_roll_joint_effort",
        "elbow_flex_joint_effort",
        "forearm_roll_joint_effort",
        "wrist_flex_joint_effort",
        "wrist_roll_joint_effort",
        "l_gripper_finger_joint_effort",
        "r_gripper_finger_joint_effort",
        "demo_cube_pose_position_x",
        "demo_cube_pose_position_y",
        "demo_cube_pose_position_z",
        "demo_cube_pose_orientation_x",
        "demo_cube_pose_orientation_y",
        "demo_cube_pose_orientation_z",
        "demo_cube_pose_orientation_w",
        "demo_cube_twist_linear_x",
        "demo_cube_twist_linear_y",
        "demo_cube_twist_linear_z",
        "demo_cube_twist_angular_x",
        "demo_cube_twist_angular_y",
        "demo_cube_twist_angular_z",
    ]

    MOTION_MASK["transport"] = [
        "l_wheel_joint_position",
        "r_wheel_joint_position",
        "l_wheel_joint_velocity",
        "r_wheel_joint_velocity",
        "l_wheel_joint_effort",
        "r_wheel_joint_effort",
        "demo_cube_pose_position_x",
        "demo_cube_pose_position_y",
        "demo_cube_pose_position_z",
        "demo_cube_pose_orientation_x",
        "demo_cube_pose_orientation_y",
        "demo_cube_pose_orientation_z",
        "demo_cube_pose_orientation_w",
        "fetch_pose_position_x",
        "fetch_pose_position_y",
        "fetch_pose_position_z",
        "fetch_pose_orientation_x",
        "fetch_pose_orientation_y",
        "fetch_pose_orientation_z",
        "fetch_pose_orientation_w",
        "demo_cube_twist_linear_x",
        "demo_cube_twist_linear_y",
        "demo_cube_twist_linear_z",
        "demo_cube_twist_angular_x",
        "demo_cube_twist_angular_y",
        "demo_cube_twist_angular_z",
        "fetch_twist_linear_x",
        "fetch_twist_linear_y",
        "fetch_twist_linear_z",
        "fetch_twist_angular_x",
        "fetch_twist_angular_y",
        "fetch_twist_angular_z",
    ]

    MOTION_MASK["place_cube"] = MOTION_MASK["pick_cube"]

    def __init__(self) -> None:
        super().__init__()

    def get_modifies(self) -> defaultdict:
        modifies = defaultdict(list)
        for variable in FETCH_FEATURE_65:
            for motion in self.MOTION_MASK:
                if variable in self.MOTION_MASK[motion]:
                    modifies[variable].append(motion)
        return modifies


class core:
    """
    paper: From Skills to Symbols: Learning Symbolic Representations for Abstract High-Level Planning

    """

    def __init__(self, motion_mask) -> None:
        self.motion_mask = motion_mask

    def _factor_o(self, motion, factors, motion_mask):
        """
        @description  : return the set of factors containing state variables that are modified by option oj as factors(oj)
        ---------
        @param  :
        -------
        @Returns  :  set of factors
        --------
        """

        res = []
        for index in factors:
            for variable in factors[index]:
                if variable in motion_mask[motion]:
                    res.append(factors[index])
                    break
        return res

    def _factor_sigma(self, sigma, factors):
        """
        @description  :  the factors over which the grounding classifier for symbol σk is defined as either factors(σk) or factors(k)
        ---------
        @param  :
        -------
        @Returns  :
        --------
        """

        res = []
        for index in factors:
            if set(sigma).issubset(factors[index]):
                res.append(factors[index])
        return res

    def _powerset(self, set_):
        power_set = [[]]
        for e in set_:
            power_set += [r + [e] for r in power_set]
        return power_set

    def _project(self, X, v):
        """
        @description  : project v out of X
        ---------
        @param  :
        ---------
        @Returns  :
        ---------
        """

        if v == None:
            return X

        project_res = []
        for e in X:
            if e not in v:
                project_res.append(e)
        return project_res

    def _is_subset(self, a, b):
        """
        @description  : check if a is a subset of b
        ---------
        @param  :
        -------
        @Returns  : True or False
        -------
        """

        for e in a:
            if e not in b:
                return False
        return True

    def _deduplication(self, _list):
        """
        @description  :  deduplication
        ---------
        @param  :
        ---------
        @Returns  :
        ---------
        """

        dedup_list = []
        for e in _list:
            if e not in dedup_list:
                dedup_list.append(e)
        return dedup_list

    def _intersection(self, a, b):
        """
        @description  :  intersection of a and b
        ---------
        @param  :
        ---------
        @Returns  :  a ^ b
        ---------
        """
        intersection_list = []
        for e in a:
            if e in b:
                intersection_list.append(e)
        intersection_set = self._deduplication(intersection_list)
        return intersection_set

    def compute_factors(self, modifies):
        F = []

        def options(fj):
            """
            @description  : return the set of options modifying the variables in factor fj
            """

            ops = []
            for variable in fj:
                ops.extend(modifies[variable])
            return set(ops)

        for state in FETCH_FEATURE_65:
            st_flag = False
            for fj in F:
                # s.t. options(fj) == modifies(si)
                if set(modifies[state]) == options(fj):
                    fj.append(state)
                    st_flag = True
                    break

            if not st_flag:
                F.append([state])

        return dict(zip(range(len(F)), F))

    def generate_symbol_set(self, factors):
        """
        @description  :  Building the Symbolic Vocabulary
        ---------
        @param  :
        ---------
        @Returns  :
        ---------
        """

        P = []
        for oi in self.motion_mask:

            # initialize
            f = self._factor_o(oi, factors, self.motion_mask)
            f_len = len(f)

            # Identify independent factors
            fi_list = copy.deepcopy(f)
            e = self.motion_mask[oi]  # e = effect(oi)
            for i in range(f_len):
                f_fi = f.copy().remove(fi_list[i])  # f_fi = f\fi
                # if fi is independent ?
                if e == self._intersection(self._project(e, fi_list[i]), self._project(e, f_fi)):
                    if len(self._project(e, f_fi)) > 0:
                        P.append(self._project(e, f_fi))
                        e = self._project(e, fi_list[i])
                        f = f.remove(fi_list[i])

            # project out all combinations of remaining factors
            for fs in self._powerset(f):
                if len(fs) > 0:
                    P.append(self._project(e, fs))

        # deduplication
        P_dedup = self._deduplication(P)
        return P_dedup

    def generate_operator_desciptors(self, motion_mask, symbol_set, factors):
        """
        @description  :  Constructing Operators
        ---------
        @param  :
        ---------
        @Returns  :
        ---------
        """

        effect_pos = defaultdict(list)
        effect_neg = defaultdict(list)
        conditional_effect = defaultdict(list)
        precondition = defaultdict(list)

        for oi in motion_mask:
            effect_pos.setdefault(oi, [])

            conditional_effect.setdefault(oi, [])
            precondition.setdefault(oi, [])

            # Direct effects
            for sigma in symbol_set:
                effect_pos[oi].append(sigma)

            # Side effects
            # Pnr = {σ|σ ∈ P, ¬refersToEffect(σr, oi)}
            Pnr = []
            for sigma in symbol_set:
                if sigma not in factors.values():
                    Pnr.append(sigma)

            effect_neg.setdefault(oi, [])
            # effects−(oi) ← {σ|σ ∈ Pnr, G(σ) ⊆ Ioi, factors(σ) ⊆ factors(oi)}
            for sigma in Pnr.copy():
                if self._is_subset(sigma, motion_mask[oi]) and self._is_subset(
                    self._factor_sigma(sigma, factors), self._factor_o(oi, factors, self.motion_mask)
                ):
                    effect_neg[oi].append(sigma)

            for sigma1 in Pnr.copy():
                for sigma2 in Pnr.copy():
                    if len(
                        self._intersection(self._factor_sigma(sigma1, factors), self._factor_o(oi, factors, self.motion_mask))
                    ) > 0 and self._project(sigma2, self._factor_o(oi, factors, self.motion_mask)):
                        conditional_effect[oi].append(sigma1)
                        conditional_effect[oi].append(sigma2)

            # # Compute preconditions
            # precondition.setdefault(motion, [])
            # for Pc in self._powerset(symbol_set):
            #     #   factors(Ioi) ⊆ ∪σ∈Pc factors(σ)
            #     factors_sigma = []
            #     ground_sigma = []
            #     # factorsOverlap = None
            #     if len(Pc) > 0:
            #         for sigma in Pc:
            #             factors_sigma.extend(self._factor(sigma, factors))
            #             if len(ground_sigma) == 0:
            #                 ground_sigma = sigma
            #             else:
            #                 ground_sigma = self._intersection(ground_sigma, sigma)

            #     condition1 = self._is_subset(self._factor(motion, factors, self.motion_mask), factors_sigma)
            #     condition2 = self._is_subset(ground_sigma, self.motion_mask[motion])
            #     # condition3 =

            #     if condition1 and condition2:
            #         precondition[motion].append(Pc)

        return effect_pos, effect_neg, conditional_effect, precondition

if __name__ == "__main__":
    mask = mask()
    core = core(mask.MOTION_MASK)
    modifies = mask.get_modifies()

    # Compute factors
    factors = core.compute_factors(modifies)
    # print(factors)

    # Generate symbol set
    symbol_set = core.generate_symbol_set(factors)
    # print(symbol_set)

    # Generate operator descriptors
    eff_pos, eff_neg, conditional_eff, preconditon = core.generate_operator_desciptors(mask.MOTION_MASK, symbol_set, factors)

    # print(eff_pos)
    # print(eff_neg)
    # print(conditional_eff)
    # print(pe)

    with open('test/baseline/eff_pos.json', 'w') as f:
        json.dump(eff_pos, f)

    # with open('test/baseline/eff_neg.json', 'w') as f:
    #     json.dump(eff_neg, f)

    # with open('test/baseline/conditional_eff.json', 'w') as f:
    #     json.dump(conditional_eff, f)
    # with open("symbol_learning/preconditions.json", "w") as f:
    #     json.dump(pre, f)