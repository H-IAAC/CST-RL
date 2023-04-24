package com.example.CSTRL.cst.behavior;

import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.cst.behavior.RL.actionManagers.ActionManager;
import com.example.CSTRL.cst.behavior.RL.actionSelectors.ActionSelector;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.FeatureExtractor;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;

import java.util.ArrayList;

public class StateActionValueFunctionRLCodelet extends RLCodelet {

    ActionManager actionManager;
    ActionSelector actionSelector;
    StateActionValueFunction stateActionValueFunction;

    public StateActionValueFunctionRLCodelet(MemoryObject perceptMO, ActionManager actionManager, ActionSelector actionSelector, StateActionValueFunction stateActionValueFunction) {
        super(perceptMO);

        this.actionManager = actionManager;
        this.actionSelector = actionSelector;
        this.stateActionValueFunction = stateActionValueFunction;
    }

    @Override
    protected void runRLStep() {
        stateActionValueFunction.update(past_state, past_action, current_state, current_reward, actionManager);
    }

    @Override
    protected ArrayList<Double> selectAction() {
        return actionSelector.selectAction(current_state, stateActionValueFunction, actionManager);
    }

    @Override
    protected void newEpisode() {
        super.newEpisode();

        actionManager.endEpisode();
        actionSelector.endEpisode();
        stateActionValueFunction.endEpisode();
    }
}
