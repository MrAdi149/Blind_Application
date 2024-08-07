plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.jetbrains.kotlin.android)
    id ("kotlin-kapt")
    id ("kotlin-android")
    id ("androidx.navigation.safeargs") version "2.7.7"
    id("de.undercouch.download") version "5.6.0"
}

apply(plugin = "androidx.navigation.safeargs")

android {
    namespace = "com.aditya.object"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.aditya.object"
        minSdk = 24
        targetSdk = 35
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
        dataBinding = true
    }

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
    implementation(project(":libuvc"))
    implementation(project(":libausbc"))
    implementation(libs.androidx.room.common)
    implementation(libs.androidx.room.ktx)

    annotationProcessor ("androidx.room:room-compiler:2.6.1")
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    kapt ("androidx.room:room-compiler:2.6.1")

    implementation(libs.firebase.crashlytics.buildtools)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

//    implementation (project(":libausbc"))
    implementation(project(":libuvc"))
//    implementation(project(":libausbc"))
//
//    implementation ("org.jitsi.react:jitsi-meet-sdk:3.10.2")
    implementation("androidx.databinding:databinding-runtime:8.5.2")

    implementation ("com.afollestad.material-dialogs:core:3.3.0")

    // App compat and UI things
    implementation ("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation ("com.google.android.material:material:1.13.0-alpha05")
    implementation ("androidx.localbroadcastmanager:localbroadcastmanager:1.1.0")
    // Navigation library
    implementation ("androidx.navigation:navigation-fragment-ktx:2.7.7")
    implementation ("androidx.navigation:navigation-ui-ktx:2.7.7")
    // CameraX core library
    implementation ("androidx.camera:camera-core:1.4.0-rc01")
    // CameraX Camera2 extensions
    implementation ("androidx.camera:camera-camera2:1.4.0-rc01")
    // CameraX Lifecycle library
    implementation ("androidx.camera:camera-lifecycle:1.4.0-rc01")
    // CameraX View class
    implementation ("androidx.camera:camera-view:1.4.0-rc01")
    //WindowManager
    implementation ("androidx.window:window:1.3.0")
    implementation ("org.tensorflow:tensorflow-lite-task-vision:0.4.4")
    // Import the GPU delegate plugin Library for GPU inference
    implementation ("org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4")
    implementation ("org.tensorflow:tensorflow-lite-gpu:2.16.1")

    implementation ("org.tensorflow:tensorflow-lite:2.16.1")
    implementation ("com.google.mlkit:text-recognition:16.0.1")
//    implementation ("com.google.mlkit:vision:19.0.2")
    implementation("com.google.mlkit:vision-common:17.3.0")

    implementation("androidx.camera:camera-mlkit-vision:1.4.0-rc01")
    // If you want to additionally use the CameraX Extensions library
    implementation("androidx.camera:camera-extensions:1.4.0-rc01")
    implementation("com.google.mlkit:text-recognition:16.0.1")

    implementation ("com.google.code.gson:gson:2.11.0")

    implementation ("com.google.android.gms:play-services-vision:20.1.3")

    implementation ("com.google.android.gms:play-services-maps:19.0.0")
    implementation ("com.google.android.gms:play-services-location:21.3.0")

    // Face features
    implementation ("com.google.mlkit:face-detection:16.1.7")


    // Or comment the dependency above and uncomment the dependency below to
    // use unbundled model that depends on Google Play Services
    implementation ("com.google.android.gms:play-services-mlkit-face-detection:17.1.0")

}

project.ext["ASSET_DIR"] = "$projectDir/src/main/assets/"
project.ext["TEST_ASSET_DIR"] = "$projectDir/src/androidTest/assets/"
apply(from = "download_tasks.gradle")
