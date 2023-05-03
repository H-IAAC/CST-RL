package com.example.CSTRL.cst.behavior;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.util.RLPercept;

import java.util.ArrayList;

abstract public class RLCodelet extends Codelet {
    protected ArrayList<Double> pastState;
    protected ArrayList<Double> pastAction;

    protected ArrayList<Double> currentState;
    protected Double reward;
    
    protected int stepCounter;

    private MemoryObject RLPerceptMO;
    private MemoryObject RLActionMO;

    public RLCodelet(MemoryObject perceptMO) {
        isMemoryObserver = true;
        perceptMO.addMemoryObserver(this);
        
        stepCounter = 0;
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

        currentState = percept.getState();

        if (pastState != null) {
            reward = percept.getReward();
            runRLStep();
        }

        pastAction = selectAction();
        RLActionMO.setI(pastAction);

        pastState = currentState;
        stepCounter++;

        endStep(percept.getEnded());
    }


    protected void newEpisode() {
        pastState = null;
        pastAction = null;
        currentState = null;
        
        stepCounter = 0;
    }

    // Update step for the RL algoritm
    abstract protected void runRLStep();

    // Returns the action that should be taken in this step
    abstract protected ArrayList<Double> selectAction();
    
    // Does any processing that needs to be done at the end of the episode, such as resetting the episode
    abstract protected void endStep(boolean episodeEnded);
}
