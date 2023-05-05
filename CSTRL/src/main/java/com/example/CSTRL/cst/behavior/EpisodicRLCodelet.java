package com.example.CSTRL.cst.behavior;

import br.unicamp.cst.core.entities.MemoryObject;

public abstract class EpisodicRLCodelet extends RLCodelet {

    public EpisodicRLCodelet(MemoryObject perceptMO) {
        super(perceptMO);
    }

    @Override
    public void endStep(boolean episodeEnded) {
        if (episodeEnded) {
            newEpisode();
        }
    }

    protected void newEpisode() {
        pastState = null;
        pastAction = null;
        currentState = null;

        stepCounter = 0;
    }
}
