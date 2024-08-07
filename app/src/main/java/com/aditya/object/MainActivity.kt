package com.aditya.`object`

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.navigation.findNavController
import com.aditya.`object`.databinding.ActivityMainBinding
import com.jiangdg.usb.USBMonitor
import com.jiangdg.uvc.UVCCamera

class MainActivity : AppCompatActivity() {


    private lateinit var activityMainBinding: ActivityMainBinding
    private lateinit var usbMonitor: USBMonitor
    private var uvcCamera: UVCCamera? = null
    private var pendingUsbDevice: UsbDevice? = null

    private val requestCameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                Toast.makeText(this, "Camera permission granted", Toast.LENGTH_LONG).show()
                setupUsbMonitor()
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_LONG).show()
            }
        }

    private val usbPermissionActionReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val action = intent.action
            if (ACTION_USB_PERMISSION == action) {
                synchronized(this) {
                    val device = intent.getParcelableExtra<UsbDevice>(UsbManager.EXTRA_DEVICE)
                    if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)) {
                        device?.let {
                            pendingUsbDevice = it
                            navigateToUsbFragment()
                        }
                    } else {
                        Log.d("MainActivity", "Permission denied for device $device")
                    }
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED) {
            setupUsbMonitor()
        } else {
            requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setupUsbMonitor() {
        usbMonitor = USBMonitor(this, object : USBMonitor.OnDeviceConnectListener {
            override fun onAttach(device: UsbDevice) {
                if (!usbMonitor.hasPermission(device)) {
                    Log.d("MainActivity", "Requesting USB permission for device: $device")
                    usbMonitor.requestPermission(device)
                } else {
                    pendingUsbDevice = device
                    navigateToUsbFragment()
                }
            }

            override fun onDetach(device: UsbDevice?) {
                device?.let {
                    Log.d("MainActivity", "USB device detached: $device")
                    uvcCamera?.stopPreview()
                    uvcCamera?.close()
                    uvcCamera = null
                }
            }

            override fun onConnect(device: UsbDevice, ctrlBlock: USBMonitor.UsbControlBlock, createNew: Boolean) {
                if (!usbMonitor.hasPermission(device)) {
                    Toast.makeText(this@MainActivity, "USB permission denied", Toast.LENGTH_SHORT).show()
                    return
                }
                pendingUsbDevice = device
                navigateToUsbFragment()
            }

            override fun onDisconnect(device: UsbDevice, ctrlBlock: USBMonitor.UsbControlBlock) {
                Log.d("MainActivity", "USB device disconnected: $device")
                uvcCamera?.stopPreview()
                uvcCamera?.close()
                uvcCamera = null
                Toast.makeText(this@MainActivity, "USB camera disconnected", Toast.LENGTH_SHORT).show()
            }

            override fun onCancel(device: UsbDevice?) {
                Toast.makeText(this@MainActivity, "USB permission denied", Toast.LENGTH_SHORT).show()
            }
        })
        usbMonitor.register()
    }

    private fun navigateToUsbFragment() {
        // Ensure navigation is done on the main thread
        Handler(Looper.getMainLooper()).post {
            findNavController(R.id.nav_host_fragment).navigate(R.id.action_camera_to_usb)
        }
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onStart() {
        super.onStart()
        val filter = IntentFilter(ACTION_USB_PERMISSION)
        registerReceiver(usbPermissionActionReceiver, filter, RECEIVER_NOT_EXPORTED)
    }
    override fun onStop() {
        super.onStop()
        unregisterReceiver(usbPermissionActionReceiver)
    }

    override fun onResume() {
        super.onResume()
        usbMonitor.register()
    }

    override fun onPause() {
        super.onPause()
        usbMonitor.unregister()
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            usbMonitor.unregister()
            uvcCamera?.destroy()
        } catch (e: Exception) {
            Log.e("MainActivity", "Error during onDestroy: ${e.message}", e)
        }
    }

    override fun onBackPressed() {
        if (Build.VERSION.SDK_INT == Build.VERSION_CODES.Q) {
            finishAfterTransition()
        } else {
            super.onBackPressed()
        }
    }

    companion object {
        const val ACTION_USB_PERMISSION = "com.aditya.object.USB_PERMISSION"
    }
}
