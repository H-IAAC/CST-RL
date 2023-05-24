package com.example.CSTRL.cst.behavior;

import br.unicamp.cst.core.entities.MemoryObject;

public abstract class EpisodicRLCodelet extends RLCodelet {

    private final int episodesPerSave = 2;
    private int episodeCounter = 0;

    public EpisodicRLCodelet(MemoryObject perceptMO) {
        super(perceptMO);
    }

    @Override
    public void endStep(boolean episodeEnded) {
        if (episodeEnded) {
            episodeCounter += 1;
            addGraphDataPoint(Integer.toString(episodeCounter));

            if (episodeCounter % episodesPerSave == 0) {
                saveGraphData();
            }

            newEpisode();
        }
    }

    protected void newEpisode() {
        pastState = null;
        pastAction = null;
        currentState = null;

        stepCounter = 0;

        cumulativeReward = 0;
    }
}
