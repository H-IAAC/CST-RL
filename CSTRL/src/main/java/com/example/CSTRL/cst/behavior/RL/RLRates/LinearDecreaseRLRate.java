package com.example.CSTRL.cst.behavior.RL.RLRates;

public class LinearDecreaseRLRate extends RLRate {
    int totalEpisodes;
    int episodesToZero;

    public LinearDecreaseRLRate(Double a, int episodesToZero) {
        super(a);

        this.episodesToZero = episodesToZero;
        this.totalEpisodes = 0;
    }

    @Override
    public void endEpisode() {
        totalEpisodes += 1;
    }

    @Override
    public Double getRate() {
        return a * (1 - Math.min(1.0, totalEpisodes / (double) episodesToZero));
    }
}
