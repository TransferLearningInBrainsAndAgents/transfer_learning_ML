using System.Collections;
using System.Collections.Generic;
using UnityEngine;


/// <summary>
/// This is a Singleton that defines the different rewards of the environment.
/// </summary>
public class ActionsRewardStructure : MonoBehaviour
{
    public static ActionsRewardStructure Instance;

    public float NotMoved;
    public float MovedForwards;
    public float MovedBack;
    public float TurnedCW;
    public float TurnedCCW;
    public float LeftPaw;
    public float RightPaw;

    private void Reset()
    {
        NotMoved = 0f;
        MovedForwards = -1f;
        MovedBack = -2f;
        TurnedCW = -1f;
        TurnedCCW = -1f;
        LeftPaw = -1f;
        RightPaw = -1f;
    }

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
            Destroy(this);
    }
}
