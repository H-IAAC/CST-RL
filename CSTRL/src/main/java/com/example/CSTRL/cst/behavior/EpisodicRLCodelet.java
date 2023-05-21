package com.example.CSTRL.cst.behavior;

import br.unicamp.cst.core.entities.MemoryObject;

public abstract class EpisodicRLCodelet extends RLCodelet {

    private final int episodesPerSave = 100;
    private int episodeCounter = 0;

    public EpisodicRLCodelet(MemoryObject perceptMO) {
        super(perceptMO);
    }

    @Override
    public void endStep(boolean episodeEnded) {
        if (episodeEnded) {
            newEpisode();

            addGraphDataPoint(Integer.toString(episodeCounter));
            if (episodeCounter % episodesPerSave == 0) {
                saveGraphData();
            }
        }
    }

    protected void newEpisode() {
        pastState = null;
        pastAction = null;
        currentState = null;

        stepCounter = 0;
        episodeCounter += 1;

        cumulativeReward = 0;
    }
}
