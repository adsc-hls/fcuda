package cetus.application;

import cetus.hir.FunctionCall;
import java.util.HashMap;
import java.util.Map;

/**
 * This class provides a way to enrol the information on the third party functions
 * which may affect the reaching definition analysis.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class ThirdPartyLibrary {

    private static Map<String, int[]> modIdxMap = new HashMap<String, int[]>();

    public static void addAll(Map<String, int[]> modIdxMap) {
        ThirdPartyLibrary.modIdxMap.putAll(modIdxMap);
    }

    public static void add(String fName, int[] modIndices) {
        ThirdPartyLibrary.modIdxMap.put(fName, modIndices);
    }

    public static boolean contains(FunctionCall fcall) {
        return (ThirdPartyLibrary.modIdxMap.get(fcall.getName().toString()) != null);
    }

    public static boolean hasSideEffectOnParameter(FunctionCall fcall) {
        if (!contains(fcall)) {
            return false;
        }
        return (modIdxMap.get(fcall.getName().toString()) != null);
    }

    public static int[] getSideEffectParamIndices(FunctionCall fcall) {
        if (hasSideEffectOnParameter(fcall) == false) {
            return null;
        }
        return modIdxMap.get(fcall.getName().toString());
    }
}
