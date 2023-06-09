package com.example.leafdiseasedetection

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.leafdiseasedetection.ml.PlantDiseaseDetection
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    private lateinit var result: TextView
    private lateinit var demoTxt: TextView
    private lateinit var classified: TextView
    private lateinit var clickHere: TextView
    private lateinit var imageView: ImageView
    private lateinit var arrowImage: ImageView
    private lateinit var picture: ImageButton
    private lateinit var browse: ImageButton

    private var imageSize = 224 // default image size

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        result = findViewById(R.id.result)
        imageView = findViewById(R.id.logo)
        picture = findViewById(R.id.button)
        browse = findViewById(R.id.button2)

        demoTxt = findViewById(R.id.demoText)
        clickHere = findViewById(R.id.click_here)
        arrowImage = findViewById(R.id.demoArrow)
        classified = findViewById(R.id.classified)

        demoTxt.visibility = View.VISIBLE
        clickHere.visibility = View.GONE
        arrowImage.visibility = View.VISIBLE
        classified.visibility = View.GONE
        result.visibility = View.GONE

        picture.setOnClickListener {
            //launch camera if we have permission
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, 1)
            } else {
                //request camera permission if we don't have
                requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
            }
        }

        browse.setOnClickListener {
            // Check if we have permission to access the gallery
            if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                val galleryIntent =
                    Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                startActivityForResult(galleryIntent, 2)
            } else {
                // Request gallery permission if we don't have
                requestPermissions(arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE), 200)
            }
        }


    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (resultCode == RESULT_OK) {
            when (requestCode) {
                1 -> { // Camera
                    val image = data?.extras?.get("data") as Bitmap
                    val dimension = min(image.width, image.height)
                    val thumbnail = ThumbnailUtils.extractThumbnail(image, dimension, dimension)
                    imageView.setImageBitmap(thumbnail)

                    demoTxt.visibility = View.GONE
                    clickHere.visibility = View.VISIBLE
                    arrowImage.visibility = View.GONE
                    classified.visibility = View.VISIBLE
                    result.visibility = View.VISIBLE

                    val scaledImage =
                        Bitmap.createScaledBitmap(thumbnail, imageSize, imageSize, false)
                     classifyImage(scaledImage)

                }

                2 -> { // Gallery
                    val selectedImageUri: Uri? = data?.data
                    val imageBitmap =
                        MediaStore.Images.Media.getBitmap(this.contentResolver, selectedImageUri)
                    val dimension = min(imageBitmap.width, imageBitmap.height)
                    val thumbnail =
                        ThumbnailUtils.extractThumbnail(imageBitmap, dimension, dimension)
                    imageView.setImageBitmap(thumbnail)

                    demoTxt.visibility = View.GONE
                    clickHere.visibility = View.VISIBLE
                    arrowImage.visibility = View.GONE
                    classified.visibility = View.VISIBLE
                    result.visibility = View.VISIBLE

                    val scaledImage =
                        Bitmap.createScaledBitmap(thumbnail, imageSize, imageSize, false)


                        classifyImage(scaledImage)


                }
            }
        }
        super.onActivityResult(requestCode, resultCode, data)
    }

    private fun classifyImage(image: Bitmap) {


        try {
            val model: PlantDiseaseDetection = PlantDiseaseDetection.newInstance(getApplicationContext())
            //create input for reference
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
            val byteBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
            byteBuffer.order(ByteOrder.nativeOrder())

            // get 1D array of 224*224 pixels in image
            val intValue = IntArray(imageSize * imageSize)
            image.getPixels(intValue, 0, image.width, 0, 0, image.width, image.height)

            // iterate over pixels and extract R, G, B values, add to bytebuffer
            var pixel = 0
            for (i in 0 until imageSize) {
                for (j in 0 until imageSize) {
                    val value = intValue[pixel++] // RGB
                    byteBuffer.putFloat(((value shr 16) and 0xFF) * (1f / 255f))
                    byteBuffer.putFloat(((value shr 8) and 0xFF) * (1f / 255f))
                    byteBuffer.putFloat((value and 0xFF) * (1f / 255f))
                }
            }

            inputFeature0.loadBuffer(byteBuffer)

            // run model interface and gets result
            val outputs: PlantDiseaseDetection.Outputs = model.process(inputFeature0)
            val outputFeatures0 = outputs.getOutputFeature0AsTensorBuffer()

            val confidence = outputFeatures0.floatArray

            // find the index of the class with the biggest confidence
            var maxPos = 0
            var maxConfidence = 0f
            for (i in confidence.indices) {
                if (confidence[i] > maxConfidence) {
                    maxConfidence = confidence[i]
                    maxPos = i
                }
            }

            val classes = arrayOf(
                "Apple: Apple scab dısease",
                "Apple: Black rot dısease",
                "Apple: Cedar apple rust dısease",
                "Apple: healthy",
                "Blueberry: healthy",
                "Cherry (ıncludıng sour): Powdery mıldew dısease",
                "Cherry (ıncludıng sour): healthy",
                "Corn (maıze): Cercospora leaf spot: Gray leaf spot dısease",
                "Corn (maıze): Common rust dısease",
                "Corn (maıze): Northern Leaf Blıght dısease",
                "Corn (maıze): healthy",
                "Grape: Black rot dısease",
                "Grape: Esca(Black Measles) dısease",
                "Grape: Leaf blıght(Isarıopsıs Leaf Spot) dısease",
                "Grape: healthy",
                "Orange: Haunglongbing(Cıtrus greenıng) dısease",
                "Peach: Bacterıal spot dısease",
                "Peach: healthy",
                "Pepper,bell: Bacterıal spot dısease",
                "Pepper,bell: healthy",
                "Potato: Early blıght dısease",
                "Potato: Late blıght dısease",
                "Potato: healthy",
                "Raspberry: healthy",
                "Soybean: healthy",
                "Squash: Powdery mıldew dısease",
                "Strawberry: Leaf scorch dısease",
                "Strawberry: healthy",
                "Tomato: Bacterial spot",
                "Tomato: Early blıght dısease",
                "Tomato: Late blıght dısease",
                "Tomato: Leaf Mold",
                "Tomato: Septorıa leaf spot dısease",
                "Tomato: Spider mıtes, Two spotted spider mıte dısease",
                "Tomato: Target Spot dısease",
                "Tomato: Tomato Yellow Leaf Curl Vırus dısease",
                "Tomato:Tomato mosaic vırus",
                "Tomato: healthy"
            )
            result.text = classes[maxPos]
            result.setOnClickListener(View.OnClickListener {
                // to search the disease on internet
                startActivity(
                    Intent(
                        Intent.ACTION_VIEW,
                        Uri.parse("https://www.google.com/search?q=" + result.text)
                    )
                )
            })

            model.close()
        } catch (e: IOException) {
            // TODO Handle the exception
        }
    }

}
