package xyz.acevedosharp;

import java.util.ArrayList;
import java.util.Arrays;

public class EpicMain {
    // minutes, repetitions, cores, runHasco, runMLPlan, datasets...
    public static void main(String[] args) {
        System.out.println("Max mem: " + Runtime.getRuntime().maxMemory());

        EpicExecutor epicExecutor = new EpicExecutor(
                new ArrayList<>(Arrays.asList(args).subList(5, args.length)),
                Integer.parseInt(args[0]),
                Integer.parseInt(args[1]),
                Integer.parseInt(args[2])
        );

        boolean runHasco = Boolean.parseBoolean(args[3]);
        boolean runMLPlan = Boolean.parseBoolean(args[4]);

        try {
            if (runHasco && runMLPlan) {
                epicExecutor.runBoth();
            } else if (runHasco) {
                epicExecutor.callHascoRunner();
            } else {
                epicExecutor.callMLPlanRunner();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
