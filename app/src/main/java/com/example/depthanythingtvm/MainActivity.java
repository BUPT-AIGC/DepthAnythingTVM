package com.example.depthanythingtvm;

import androidx.appcompat.app.AppCompatActivity;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.graphics.Bitmap;
import android.widget.ImageView;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.chaquo.python.PyObject;

import org.apache.tvm.Function;
import org.apache.tvm.ArgTypeCode;
import org.apache.tvm.Module;
import org.apache.tvm.NDArray;
import org.apache.tvm.Device;
import org.apache.tvm.TVMValue;
import org.apache.tvm.TVMType;
import org.apache.tvm.contrib.GraphModule;
//import org.apache.tvm.Base;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.io.IOException;
import java.util.List;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

import java.io.OutputStream;


public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    AssetManager assetManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        assetManager = getAssets();
        System.loadLibrary("tvm4j_runtime_packed");

        // 初始化Python环境
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        // 运行处理流程
        try {
            processImage();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void processImage() throws IOException {
        String imagePath = copyAssetToInternalStorage("demo02.jpg");
        Python py = Python.getInstance();
        PyObject pyModule = py.getModule("process");

        // 调用pre_process
        PyObject preProcessResult = pyModule.callAttr("pre_process", imagePath);
        // 解析返回值
        List<PyObject> resultList = preProcessResult.asList();

        // 获取img_pad
        PyObject imgPadPyArray = resultList.get(0);
        float[] imgPadArray = imgPadPyArray.toJava(float[].class);

        // 获取尺寸参数
        int h_new = resultList.get(1).toInt();
        int w_new = resultList.get(2).toInt();
        int h = resultList.get(3).toInt();
        int w = resultList.get(4).toInt();

        // 调用tvm_inference
        float[] outputArray = tvm_inference(imgPadArray);
        PyObject outputPyArray = py.getModule("numpy").callAttr("array", outputArray);

        // 调用post_process
        PyObject postProcessResult = pyModule.callAttr("post_process", outputPyArray, h_new, w_new, h, w);

        // 处理post_process的返回值
        List<PyObject> postResultList = postProcessResult.asList();

        // 获取depth_colormap列表
        PyObject depthColormapPyArray = postResultList.get(0);
        int[] depthColormapArray = depthColormapPyArray.toJava(int[].class);

        // 获取图像的shape
        List<PyObject> shapeList = postResultList.get(1).asList();
        int height = shapeList.get(0).toInt();
        int width = shapeList.get(1).toInt();

        // 创建目标Bitmap
        Bitmap resultBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        resultBitmap.setPixels(depthColormapArray, 0, width, 0, 0, width, height);

        // 更新UI
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                // 显示resultBitmap
                ImageView imageView = findViewById(R.id.imageView);
                imageView.setImageBitmap(resultBitmap);
            }
        });
    }

    private float[] tvm_inference(float[] img) throws IOException {
        ////// Read complied model
        String libFilename = "compiled_model_android_with_params.so";
        byte[] modelLibByte = getBytesFromFile(assetManager, libFilename);

        ////// Specify device
        Device cpuDev = Device.cpu();

        ////// Upload complied model on application cache folder
        String libPath = getTempLibFilePath(libFilename);
        FileOutputStream fos = new FileOutputStream(libPath);
        fos.write(modelLibByte);
        fos.close();

        ////// Load complied module
        Module modelLib = null;
        try {
            modelLib = Module.load(libPath);
        } catch (Exception e) {
            Log.e(TAG, "Failed to load model library", e);
        }

        GraphModule runtime = null;
        try {
//            long device_64 = deviceToInt64(cpuDev);
//            Base._LIB.tvmFuncPushArgHandle(device_64, ArgTypeCode.DLDEVICE.ordinal());
//            TVMValue ret = packed_fn.call();
//            Module mod = ret.asModule();
//            runtime = new GraphModule(mod, cpuDev);

            Module mod = modelLib.getFunction("default").call(cpuDev).asModule();
            runtime = new GraphModule(mod, cpuDev);

        } catch (Exception e) {
            Log.e(TAG, "Failed to call", e);
        }

        ////// Load input data into runtime
        NDArray dev_data = NDArray.empty(new long[]{1, 3, 518, 518}, new TVMType("float32"), cpuDev);
        dev_data.copyFrom(img);
        // set input
        runtime.setInput("input", dev_data);

        ////// Runtime go run
        try {
            runtime.run();
        } catch (Exception e) {
            Log.e(TAG, "Failed to run", e);
            throw new RuntimeException("Failed to run inference");
        }

        ////// Receive output from runtime
        NDArray output = NDArray.empty(new long[]{1, 518, 518}, new TVMType("float32"), cpuDev);
        runtime.getOutput(0, output);
        return output.asFloatArray();
    }

    private String copyAssetToInternalStorage(String filename) {
        File outFile = new File(getFilesDir(), filename);
        try (InputStream in = assetManager.open(filename);
             OutputStream out = new FileOutputStream(outFile)) {
            byte[] buffer = new byte[1024];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
            out.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return outFile.getAbsolutePath();
    }

    private byte[] getBytesFromFile(AssetManager assets, String fileName) throws IOException {
        InputStream is = assets.open(fileName);
        int length = is.available();
        byte[] bytes = new byte[length];
        // Read in the bytes
        int offset = 0;
        int numRead = 0;
        try {
            while (offset < bytes.length
                    && (numRead = is.read(bytes, offset, bytes.length - offset)) >= 0) {
                offset += numRead;
            }
        } finally {
            is.close();
        }
        // Ensure all the bytes have been read in
        if (offset < bytes.length) {
            throw new IOException("Could not completely read file " + fileName);
        }
        return bytes;
    }

    private final String getTempLibFilePath(String fileName) throws IOException {
        File tempDir = File.createTempFile("tvm4j_demo_", "");
        if (!tempDir.delete() || !tempDir.mkdir()) {
            throw new IOException("Couldn't create directory " + tempDir.getAbsolutePath());
        }
        return (tempDir + File.separator + fileName);
    }

    public static long deviceToInt64(Device dev) {
        // Create a ByteBuffer with native byte order
        ByteBuffer buffer = ByteBuffer.allocate(8).order(ByteOrder.nativeOrder());

        // Put device_type and device_id into the buffer
        buffer.putInt(dev.deviceType);
        buffer.putInt(dev.deviceId);

        // Reset the buffer position to the beginning
        buffer.flip();

        // Read the buffer as a long (64-bit integer)
        return buffer.getLong();
    }
}