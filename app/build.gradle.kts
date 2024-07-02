plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.jetbrains.kotlin.android)
    id ("kotlin-kapt")
    id ("kotlin-android")
    id ("androidx.navigation.safeargs") version "2.7.7"
    id("de.undercouch.download") version "5.6.0"
}

apply(plugin= "androidx.navigation.safeargs")

android {
    namespace = "com.aditya.object"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.aditya.object"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    buildFeatures {
        viewBinding = true
    }
//    androidResources {
//        noCompress = 'tflite'
//    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // App compat and UI things
    implementation ("androidx.lifecycle:lifecycle-runtime-ktx:2.8.3")
    implementation ("com.google.android.material:material:1.12.0")
    implementation ("androidx.localbroadcastmanager:localbroadcastmanager:1.1.0")

    // Navigation library
    implementation ("androidx.navigation:navigation-fragment-ktx:2.7.7")
    implementation ("androidx.navigation:navigation-ui-ktx:2.7.7")

    // CameraX core library
    implementation ("androidx.camera:camera-core:1.3.4")

    // CameraX Camera2 extensions
    implementation ("androidx.camera:camera-camera2:1.3.4")

    // CameraX Lifecycle library
    implementation ("androidx.camera:camera-lifecycle:1.3.4")

    // CameraX View class
    implementation ("androidx.camera:camera-view:1.3.4")

    //WindowManager
    implementation ("androidx.window:window:1.3.0")

    implementation ("org.tensorflow:tensorflow-lite-task-vision:0.4.0")
    // Import the GPU delegate plugin Library for GPU inference
    implementation ("org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4")
    implementation ("org.tensorflow:tensorflow-lite-gpu:2.10.0")

    implementation ("com.google.code.gson:gson:2.10.1")
}

project.ext["ASSET_DIR"] = "$projectDir/src/main/assets/"
project.ext["TEST_ASSET_DIR"] = "$projectDir/src/androidTest/assets/"
apply(from = "download_tasks.gradle")