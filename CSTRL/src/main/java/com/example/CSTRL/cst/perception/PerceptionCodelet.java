package com.example.CSTRL.cst.perception;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.util.RLPercept;

public class PerceptionCodelet extends Codelet {
    private MemoryObject stateMO;
    private MemoryObject perceptMO;

    private boolean lastEnded = false;

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
        perceptMO.setI(stateMO.getI());
    }
}
