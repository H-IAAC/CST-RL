package com.example.CSTRL.cst.perception;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.util.RLPercept;

public class PerceptionCodelet extends Codelet {
    private MemoryObject stateMO;
    private MemoryObject perceptMO;

    private int cycleCount = 0;
    private final int cyclesPerUpdate = 10;

    @Override
    public void accessMemoryObjects() {
        stateMO = (MemoryObject) getInput("STATE");
        perceptMO = (MemoryObject) getOutput("RLPERCEPT");
    }

    @Override
    public void calculateActivation() {

    }

    @Override
    public void proc() {
        cycleCount += 1;
        if (cycleCount > cyclesPerUpdate || ((RLPercept) perceptMO.getI()).getEnded()) {
            cycleCount = 0;

            perceptMO.setI(stateMO.getI());
        }
    }
}
