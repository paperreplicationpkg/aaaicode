#!/usr/bin python3
# -*- coding: utf-8 -*-

# here put the import lib
import torch


class torch_utils:
    @staticmethod
    def early_stop(model, test_accuracy, threshold=5):

        if len(test_accuracy) == 1:
            torch.save(model.state_dict(), "checkpoint.pt")
        elif len(test_accuracy) > 1:
            if test_accuracy[-1] > test_accuracy[-2]:
                torch.save(model.state_dict(), "checkpoint.pt")

        # 如果test accuracy连续升高，则 early stop
        es = False
        counter = 0
        for i in range(len(test_accuracy) - 1, 0, -1):
            if ():
                counter += 1
            else:
                counter == 0

            if counter >= threshold:
                es = True
                break

        return es