using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;


public class ExploreCornersAreaRewards : MonoBehaviour
{
    RewardController rewardsController;
    Dictionary<string, bool> areasTouched;
    Dictionary<string, float> timers;
    Dictionary<string, string> oppositeAreas;
    Dictionary<string, float> inactivityTimes;

    public float rewAreaMinimumInactivity = 1.0f;
    public float rewAreaMaximumInactivity = 10.0f;

    void Start()
    {
        rewardsController = this.GetComponent<RewardController>();

        EventManager.Instance.onBodyCollisionInArea.AddListener(BodyCollidedWithReweardArea);

        areasTouched = new Dictionary<string, bool>{
            {"AreaHigh_Left",true},
            {"AreaMedium_Left", true},
            {"AreaHigh_Right", true},
            {"AreaMedium_Right", true},
            {"AreaHigh_Front", true},
            {"AreaMedium_Front", true},
            {"AreaHigh_Back", true},
            {"AreaMedium_Back", true}
        };

        timers = new Dictionary<string, float>
        {
            {"AreaHigh_Left",0.0f},
            {"AreaMedium_Left", 0.0f},
            {"AreaHigh_Right", 0.0f},
            {"AreaMedium_Right", 0.0f},
            {"AreaHigh_Front", 0.0f},
            {"AreaMedium_Front", 0.0f},
            {"AreaHigh_Back", 0.0f},
            {"AreaMedium_Back", 0.0f}
        };

        oppositeAreas = new Dictionary<string, string>
        {
            {"AreaHigh_Left","AreaHigh_Right"},
            {"AreaMedium_Left", "AreaMedium_Right"},
            {"AreaHigh_Right", "AreaHigh_Left"},
            {"AreaMedium_Right", "AreaMedium_Left"},
            {"AreaHigh_Front", "AreaHigh_Back"},
            {"AreaMedium_Front", "AreaMedium_Back"},
            {"AreaHigh_Back", "AreaHigh_Front"},
            {"AreaMedium_Back", "AreaMedium_Front"}
        };

        //rewAreaMinimumInactivity = (float)(Variables.Saved.Get("rewAreaMinimumInactivity"));
        //rewAreaMaximumInactivity = (float)(Variables.Saved.Get("rewAreaMaximumInactivity"));

        inactivityTimes = new Dictionary<string, float>
        {
            {"AreaHigh_Left", 0.0f},
            {"AreaMedium_Left", 0.0f},
            {"AreaHigh_Right", 0.0f},
            {"AreaMedium_Right", 0.0f},
            {"AreaHigh_Front", 0.0f},
            {"AreaMedium_Front", 0.0f},
            {"AreaHigh_Back", 0.0f},
            {"AreaMedium_Back", 0.0f}
        };
    }

    private void Update()
    {
        foreach (string area_names in areasTouched.Keys)
        {
            timers[area_names] += Time.deltaTime;
            
        }
        
    }

    void BodyCollidedWithReweardArea(string rewardArea)
    {
        
        Debug.Log($"{rewardArea} touched  = {areasTouched[rewardArea]}");
        Debug.Log($"Opposite {oppositeAreas[rewardArea]} touched = {areasTouched[oppositeAreas[rewardArea]]}");
        Debug.Log($"timer = {timers[rewardArea]}");
        Debug.Log($"Inactivity = {inactivityTimes[rewardArea]}");
        
        if (areasTouched[oppositeAreas[rewardArea]] && timers[rewardArea] >= inactivityTimes[rewardArea]) 
        {
            Debug.Log("Reward given");
            areasTouched[oppositeAreas[rewardArea]] = false;
            areasTouched[rewardArea] = true;
            inactivityTimes[rewardArea] = Random.Range(rewAreaMinimumInactivity, rewAreaMaximumInactivity);
            timers[rewardArea] = 0.0f;

            rewardsController.SaveNewRewardDueToAreaTouched(rewardArea);
        }
        
    }
}
