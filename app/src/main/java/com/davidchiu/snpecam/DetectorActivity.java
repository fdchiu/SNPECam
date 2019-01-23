/*
 *  Created by David Chiu
 *  Dec. 28th, 2018 based on tensorflow's original file in android sample
 *
 */


/*
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.davidchiu.snpecam;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.text.TextUtils;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import java.util.List;
import java.util.Vector;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";
  
  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;

  private static final boolean MAINTAIN_ASPECT = false;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Ncnn detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  //private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  public static final String NCNN_PARAM_FILE = "mobilenet_yolo.param.bin";
    public static final String NCNN_WEIGHTS_FILE = "mobilenet_yolo.bin";
    public static final String NCNN_LABEL_FILE = "labelcoco20.txt";
    private String ncnnParamFile;
    private String ncnnWeightsFile;
    private String ncnnLabelFile;

    public static final String MODEL_FIEL = "mobilenet_ssd.dlc";
    public static final String LABEL_FILE = "labels.txt";

    private String modelFile;
    private  String labelFile;
    private SNPENet snpenet;



   public static final int NCNN_YOLO_WIDTH = 416;
    public static final int NCNN_YOLO_HEIGHT = 416;
    public int frameWidth ;
    public int frameHeight;
    private Matrix frameToCanvasMatrix;
    public List<Detection> detections;
    private final Paint boxPaint = new Paint();
    private  BorderedText borderedText=null;
    public int cropWidth;          //width and height for image input to model
    public int cropHeight;

    //private BorderedText borderedText;
  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
      if (BuildConfig.model_file != null) {
          modelFile = BuildConfig.model_file;
      } else {
          modelFile = MODEL_FIEL;
      }
      if (BuildConfig.label_file != null) {
          labelFile = BuildConfig.label_file;
      } else {
          labelFile = LABEL_FILE;
      }


      final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

      boxPaint.setColor(Color.RED);
      boxPaint.setStyle(Style.STROKE);
      boxPaint.setStrokeWidth(12.0f);
      boxPaint.setStrokeCap(Paint.Cap.ROUND);
      boxPaint.setStrokeJoin(Paint.Join.ROUND);
      boxPaint.setStrokeMiter(100);

    //tracker = new MultiBoxTracker(this);

      cropWidth = NCNN_YOLO_WIDTH;     //was: 128
      cropHeight = NCNN_YOLO_HEIGHT;    //was: 96

      if (BuildConfig.cropW != 0) {
          cropWidth = BuildConfig.cropW;
      }
      if (BuildConfig.cropH != 0) {
          cropHeight = BuildConfig.cropH;
      }

      previewWidth = size.getWidth();
      previewHeight = size.getHeight();
      frameHeight = previewHeight;
      frameWidth = previewWidth;

if(false) {
    try {
        detector = new Ncnn();
        detector.setImageSize(size.getWidth(), size.getHeight());
        detector.initNcnn(this, ncnnParamFile, ncnnWeightsFile, ncnnLabelFile);
        //cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final Exception e) {
        //LOGGER.e("Exception initializing classifier!", e);
        Toast toast =
                Toast.makeText(
                        getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
        toast.show();
        finish();
    }
}
    snpenet = new SNPENet();
      snpenet.setImageSize(size.getWidth(), size.getHeight());
      snpenet.setModelName(BuildConfig.model);
    snpenet.init(this, modelFile, labelFile);

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);

    if (false) {
        croppedBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Config.ARGB_8888);
        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropWidth, cropHeight,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);
    } else {
        prepare(sensorOrientation);  //90 means vertical screen while 0 means horizontal
    }
    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new OverlayView.DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            draw(canvas);
            if (isDebug()) {
              //tracker.drawDebug(canvas);
            }
          }
        });

    addCallback(
        new OverlayView.DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            if (!isDebug()) {
              return;
            }
            final Bitmap copy = cropCopyBitmap;
            if (copy == null) {
              return;
            }

            final int backgroundColor = Color.argb(100, 0, 0, 0);
            canvas.drawColor(backgroundColor);

            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                canvas.getWidth() - copy.getWidth() * scaleFactor,
                canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (detector != null) {
              final String statString =""; //detector.getStatString();
              final String[] statLines = statString.split("\n");
              for (final String line : statLines) {
                lines.add(line);
              }
            }
            lines.add("");

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
          }
        });
  }

  OverlayView trackingOverlay;

    boolean loadTestImage=false;

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
      trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
        Log.i(this.getClass().getSimpleName(), " detect drop frame: " + timestamp);
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    if (loadTestImage) {
        rgbFrameBitmap = loadTestImage(null);
        if (rgbFrameBitmap == null) {
            readyForNextImage();
            computingDetection = false;
            return;
        }
    } else {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    }

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            //final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
              //detections = detector.detect(croppedBitmap);
              detections = snpenet.run(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
              Log.i("snpecam", " detect : " + lastProcessingTimeMs);
            if (detections != null) {
                Log.i("detect: ", " objects: " + detections.size());
            }
         if(true) {
             cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
             final Canvas canvas = new Canvas(cropCopyBitmap);
             final Paint paint = new Paint();
             paint.setColor(Color.RED);
             paint.setStyle(Style.STROKE);
             paint.setStrokeWidth(2.0f);

             requestRender();
         }
            computingDetection = false;
          }
        });
  }

    public synchronized void draw(final Canvas canvas) {
        final boolean rotated = sensorOrientation % 180 == 90;
        final float multiplier =
                Math.min(canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
                        canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
        frameToCanvasMatrix =
                ImageUtils.getTransformationMatrix(
                        frameWidth,
                        frameHeight,
                        (int) (multiplier * (rotated ? frameHeight : frameWidth)),
                        (int) (multiplier * (rotated ? frameWidth : frameHeight)),
                        sensorOrientation,
                        false);
        if (detections == null) return;
        for (final Detection recognition : detections) {
            final RectF trackedPos = new RectF(recognition.location);

            frameToCanvasMatrix.mapRect(trackedPos);
            boxPaint.setColor(recognition.color);

            final float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
            canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

            final String labelString =
                    !TextUtils.isEmpty(recognition.title)
                            ? String.format("%s %.2f", recognition.title, recognition.detectionConfidence)
                            : String.format("%.2f", recognition.detectionConfidence);
            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.bottom, labelString);
        }
    }

    public void prepare(int sensorOrientation) {
        if (sensorOrientation == 0 || sensorOrientation == 180) {
                croppedBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Bitmap.Config.ARGB_8888);
                frameToCropTransform =
                        ImageUtils.getTransformationMatrix(
                                previewWidth, previewHeight,
                                cropWidth, cropHeight,
                                sensorOrientation, MAINTAIN_ASPECT); // from [previewWidth, previewHeight] to [cropW, cropH]

        } else {
                croppedBitmap = Bitmap.createBitmap(cropHeight, cropWidth, Bitmap.Config.ARGB_8888);
                frameToCropTransform =
                        ImageUtils.getTransformationMatrix(
                                previewWidth, previewHeight,
                                cropHeight, cropWidth,
                                sensorOrientation, MAINTAIN_ASPECT); // from [previewWidth, previewHeight] to [cropW, cropH]
        }

            cropToFrameTransform = new Matrix();
            frameToCropTransform.invert(cropToFrameTransform);
    }


  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onSetDebug(final boolean debug) {
    //detector.enableStatLogging(debug);
  }

  @Override
  public void onDestroy() {
      super.onDestroy();
      if (snpenet != null) {
          snpenet.destroy();
          snpenet = null;
      }
  }

    /**
     * This method is for loading test images
     */
    protected int _testcount = -1;
    protected int _testtotal = 1;
    protected String _imagePath = "/sdcard";
    protected String _imagename = "test";
    protected String _imageextension = "jpg";

    protected Bitmap loadTestImage(String path) {
        if (path != null) {
            _imagePath = path;
        }
        if (_imagePath == null || _imageextension == null || _imagename == null) {
            throw new RuntimeException("path/file name/extension not set");
        }
        _testcount++;
        if (_testcount >= _testtotal) {
            _testcount = 0;
        }
        String _file_path_name = _imagePath + "/" + _imagename + _testcount + "." + _imageextension;
        Log.i("loadTestImage", _file_path_name);
        return BitmapFactory.decodeFile(_file_path_name);
    }

}
