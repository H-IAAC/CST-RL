package com.example.CSTRL.cst.behavior;

import br.unicamp.cst.core.entities.MemoryObject;

import java.util.ArrayList;
import java.util.Arrays;

public class TestRLCodelet extends RLCodelet {
    private Double t = 0.0;
    private Double mod = 0.1;

    public TestRLCodelet(MemoryObject perceptMO) {
        super(perceptMO);
    }

    @Override
    protected void runRLStep() {
        t += mod;

        if (t >= 1.0 || t <= 0.0) {
            mod = -mod;
        }
    }

    @Override
    protected ArrayList<Double> selectAction() {
        return new ArrayList<Double>(Arrays.asList(t, 0.0, 0.0, 0.0));
    }
}
