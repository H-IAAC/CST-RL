package com.example.CSTRL.cst.behavior;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.util.RLPercept;

import java.util.ArrayList;

abstract public class RLCodelet extends Codelet {
    protected ArrayList<Double> past_state;
    protected ArrayList<Double> past_action;

    protected ArrayList<Double> current_state;
    protected Double reward;

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

        current_state = percept.getState();

        if (past_state != null) {
            reward = percept.getReward();
            runRLStep();
        }

        past_action = selectAction();
        RLActionMO.setI(past_action);

        if (percept.getEnded()) {
            newEpisode();
        } else {
            past_state = current_state;
        }
    }


    protected void newEpisode() {
        past_state = null;
        past_action = null;
        current_state = null;
    }

    // Update step for the RL algoritm
    abstract protected void runRLStep();

    // Returns the action that should be taken in this step
    abstract protected ArrayList<Double> selectAction();
}
