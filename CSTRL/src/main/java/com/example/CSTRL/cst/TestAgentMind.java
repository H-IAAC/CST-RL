package com.example.CSTRL.cst;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.cst.behavior.TestRLCodelet;

public class TestAgentMind extends AgentMind {

    @Override
    protected Codelet getRLCodelet(MemoryObject perceptMO) {
        return new TestRLCodelet(perceptMO);
    }
}
