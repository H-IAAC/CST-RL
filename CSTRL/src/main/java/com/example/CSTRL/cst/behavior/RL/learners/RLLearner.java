package com.example.CSTRL.cst.behavior.RL.learners;

import com.example.CSTRL.cst.behavior.RL.actionSpaces.ActionSpace;
import com.example.CSTRL.util.RLPercept;

import java.util.ArrayList;

public abstract class RLLearner {
    protected ActionSpace actionSpace;

    public void setActionSpace(ActionSpace actionSpace) {
        this.actionSpace = actionSpace;
    }

    abstract public void rlStep(ArrayList<RLPercept> trial);

    abstract public ArrayList<Double> selectAction(ArrayList<Double> s);

    abstract public void endEpisode();
}
