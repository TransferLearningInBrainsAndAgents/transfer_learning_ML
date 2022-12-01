using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PostProcess : MonoBehaviour
{

    private Material _material;
    public Shader _shader;

    //public Renderer screenGrabRenderer;
    private Texture2D destinationTexture;
    private Color32[] pixels;
    private static byte[] pixelsComp;

    private void Awake()
    {
        Application.targetFrameRate = 0;
        QualitySettings.vSyncCount = 0;
    }

    // Start is called before the first frame update
    void Start()
    {
        _material = new(_shader);

        destinationTexture = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);

        var buffer_size = Screen.width * Screen.height;
        pixelsComp = new byte[buffer_size];

    }


    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        Graphics.Blit(source, destination, _material);
    }


}
