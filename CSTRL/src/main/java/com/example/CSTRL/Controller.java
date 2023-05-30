package com.example.CSTRL;

import com.example.CSTRL.cst.*;
import com.example.CSTRL.util.*;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.Arrays;

@RestController
public class Controller {
    private AgentMind agentMind;

    // Initialize
    @GetMapping("/initialize")
    public GodotContainer initialize() {
        agentMind = new SimplifiedLFAAgentMind();

        return new GodotContainer(ReturnType.INIT);
    }

    // Receive percepts
    @PostMapping("/sendpercept")
    public GodotContainer receivePercept(@RequestBody RLPercept percept) {
        if (agentMind != null) {
            agentMind.setState(percept);
        }

        if (percept.getEnded()) {
            return new GodotContainer(ReturnType.RESET);
        }

        return new GodotContainer(ReturnType.PERCEPT);
    }

    // Send actions
    @GetMapping("/getaction")
    public RLAction sendAction() {
        if (agentMind != null) {
            return new RLAction(agentMind.getAction());
        }
        return new RLAction(new ArrayList<Double>());

    }

    // Test get mapping
    @GetMapping("/gettest")
    public TestContainer getTest(@RequestParam(defaultValue = "NAN") String t) {
        return new TestContainer("API is running!", t);
    }

    // Test post mapping
    @PostMapping("/posttest")
    public TestContainer postTest(@RequestBody TestContainer testContainer) {
        return new TestContainer("API is running! - " + testContainer.getMessage(), testContainer.getAddon());
    }

    // Type test
    @PostMapping("/typetest")
    public TypeTestContainer typeTest(@RequestBody TypeTestContainer typeTestContainer) {
        return new TypeTestContainer("API is running! - " + typeTestContainer.getMessage() + " - " + typeTestContainer.getAddon(), typeTestContainer.getArray().toString() + " - " + Integer.toString(typeTestContainer.getAnInt()), typeTestContainer.getArray(), typeTestContainer.getAnInt());
    }
}
