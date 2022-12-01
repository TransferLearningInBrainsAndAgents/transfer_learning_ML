

using System.Collections;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

public class CameraObservations : MonoBehaviour
{
    private GraphicsFormat format;

    
    IEnumerator Start()
    {
        yield return new WaitForSeconds(1);

        while (true)
        {
            yield return new WaitForSeconds(0.001f);
            //yield return new WaitForEndOfFrame();

            var rt = RenderTexture.GetTemporary(Screen.width, Screen.height, 32);

            format = rt.graphicsFormat;
            
            ScreenCapture.CaptureScreenshotIntoRenderTexture(rt);

            AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, OnCompleteReadback);

            RenderTexture.ReleaseTemporary(rt);
        }
    }


    void OnCompleteReadback(AsyncGPUReadbackRequest request)
    {
        if (request.hasError)
        {
            Debug.Log("GPU readback error detected.");
            return;
        }

        byte[] array = request.GetData<byte>().ToArray();

        byte[] pngBytes = ImageConversion.EncodeArrayToPNG(array, format, (uint)Screen.width, (uint)Screen.height);

        EventManager.Instance.onObservationReady.Invoke(pngBytes);

        /*
        Task.Run(() =>
        {
            File.WriteAllBytes($"E:/Temp/test.png", pngBytes);
        });
        */


    }
}