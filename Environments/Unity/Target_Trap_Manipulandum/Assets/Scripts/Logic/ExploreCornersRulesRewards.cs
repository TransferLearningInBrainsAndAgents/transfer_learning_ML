using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
using UnityEditorInternal;
#endif

public class ExploreCornersRulesRewards : MonoBehaviour
{
    RewardController rewardsController;
    Dictionary<string, bool> areasTouched;
    Dictionary<string, float> timers;
    Dictionary<string, string> oppositeAreas;
    Dictionary<string, float> inactivityTimes;


    public RuleNamestoIndicies ruleNamesToIndicies;
    [System.Serializable]
    public class RuleNamestoIndicies : UDictionary<string, int> { }

    private Dictionary<int, string> ruleIndiciesToNames;

    [ExecuteInEditMode]
    void OnValidate()
    {
        ruleNamesToIndicies["AreaHigh_Left"] = 0;
        ruleNamesToIndicies["AreaMedium_Left"] = 1;
        ruleNamesToIndicies["AreaHigh_Right"] = 2;
        ruleNamesToIndicies["AreaMedium_Right"] = 3;
        ruleNamesToIndicies["AreaHigh_Front"] = 4;
        ruleNamesToIndicies["AreaMedium_Front"] = 5;
        ruleNamesToIndicies["AreaHigh_Back"] = 6;
        ruleNamesToIndicies["AreaMedium_Back"] = 7;

        ruleIndiciesToNames = new Dictionary<int, string>
        {
            {0, "AreaHigh_Left"},
            {1, "AreaMedium_Left"},
            {2, "AreaHigh_Right"},
            {3, "AreaMedium_Right"},
            {4, "AreaHigh_Front"},
            {5, "AreaMedium_Front"},
            {6, "AreaHigh_Back"},
            {7, "AreaMedium_Back"}
        };
    }

    void Start()
    {
        EventManager.Instance.onBodyCollisionInArea.AddListener(BodyCollidedWithReweardArea);
    }

    

    void BodyCollidedWithReweardArea(string rewardArea)
    {
        /*
        Debug.Log($"{rewardArea} touched  = {areasTouched[rewardArea]}");
        Debug.Log($"Opposite {oppositeAreas[rewardArea]} touched = {areasTouched[oppositeAreas[rewardArea]]}");
        Debug.Log($"timer = {timers[rewardArea]}");
        Debug.Log($"Inactivity = {inactivityTimes[rewardArea]}");
        */
       
        int ruleIndex = ruleNamesToIndicies[rewardArea];
        int[] rulesUpdate = { 0, 0, 0, 0, 0, 0, 0, 0 };
        rulesUpdate[ruleIndex] = 1;
        EventManager.Instance.onRewardRulesUpdate.Invoke(rulesUpdate);
    }
}
