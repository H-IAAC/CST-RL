package com.example.CSTRL.cst.behavior;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.util.RLPercept;

import java.util.ArrayList;

abstract public class RLCodelet extends Codelet {
    protected ArrayList<Double> past_state;
    protected ArrayList<Double> past_action;
    protected Double past_reward;

    protected ArrayList<Double> current_state;
    protected Double current_reward;

    private MemoryObject RLPerceptMO;
    private MemoryObject RLActionMO;

    public RLCodelet(MemoryObject perceptMO) {
        isMemoryObserver = true;
        perceptMO.addMemoryObserver(this);
    }

    @Override
    public void accessMemoryObjects() {
        RLPerceptMO = (MemoryObject) getInput("RLPERCEPT");
        RLActionMO = (MemoryObject) getOutput("RLACTION");
    }

    @Override
    public void calculateActivation() {

    }

    @Override
    public void proc() {
        RLPercept percept = (RLPercept) RLPerceptMO.getI();

        past_state = current_state;
        past_reward = current_reward;

        current_state = percept.getState();
        current_reward = percept.getReward();

        if (past_state != null) {
            runRLStep();

            if (percept.getEnded()) {
                newEpisode();
            }
        }

        past_action = selectAction();
        RLActionMO.setI(past_action);
    }


    protected void newEpisode() {
        past_state = null;
        past_reward = null;
        past_action = null;
        current_state = null;
        current_reward = null;
    }

    // Update step for the RL algoritm
    abstract protected void runRLStep();

    // Returns the action that should be taken in this step
    abstract protected ArrayList<Double> selectAction();
}
