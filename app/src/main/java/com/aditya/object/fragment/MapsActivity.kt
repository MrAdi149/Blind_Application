package com.aditya.`object`.fragment

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.Resources
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import androidx.core.app.ActivityCompat
import android.content.Intent
import android.location.Geocoder
import android.net.Uri
import android.speech.tts.TextToSpeech
import android.widget.Toast
import com.aditya.`object`.R
import com.aditya.`object`.databinding.ActivityMapsBinding
import com.aditya.`object`.fragment.CameraFragment
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.SupportMapFragment
import com.google.android.gms.maps.model.*
import java.io.IOException
import java.util.*

class MapsActivity : AppCompatActivity(), OnMapReadyCallback {
    private val TAG = MapsActivity::class.java.simpleName
    private val REQUEST_LOCATION_PERMISSION = 1
    private lateinit var map: GoogleMap
    private lateinit var binding: ActivityMapsBinding
    private lateinit var textToSpeech: TextToSpeech
    private var currentLatLng: LatLng? = null
    private var destinationLatLng: LatLng? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMapsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val mapFragment = supportFragmentManager
            .findFragmentById(R.id.map) as SupportMapFragment
        mapFragment.getMapAsync(this)

        textToSpeech = TextToSpeech(this) { status ->
            if (status != TextToSpeech.ERROR) {
                textToSpeech.language = Locale.US
            }
        }

        val latitude = intent.getDoubleExtra("CURRENT_LAT", 0.0)
        val longitude = intent.getDoubleExtra("CURRENT_LNG", 0.0)
        currentLatLng = LatLng(latitude, longitude)

        val destLat = intent.getDoubleExtra("DESTINATION_LAT", 0.0)
        val destLng = intent.getDoubleExtra("DESTINATION_LNG", 0.0)
        destinationLatLng = LatLng(destLat, destLng)

        Log.d(TAG, "Current Location: ${currentLatLng?.latitude}, ${currentLatLng?.longitude}")
        Log.d(TAG, "Destination Location: ${destinationLatLng?.latitude}, ${destinationLatLng?.longitude}")

        if (destinationLatLng != null) {
            openGoogleMapsForNavigation(destinationLatLng!!)
        }
    }

    override fun onMapReady(googleMap: GoogleMap) {
        map = googleMap

        val overlaySize = 100f
        val androidOverlay = currentLatLng?.let {
            GroundOverlayOptions()
                .image(BitmapDescriptorFactory.fromResource(R.drawable.blindicon))
                .position(it, overlaySize)
        }

        val zoomLevel = 15f

        currentLatLng?.let { CameraUpdateFactory.newLatLngZoom(it, zoomLevel) }
            ?.let { map.moveCamera(it) }
        map.addMarker(MarkerOptions().position(currentLatLng!!))
        if (androidOverlay != null) {
            map.addGroundOverlay(androidOverlay)
        }

        setMapLongClick(map)
        setPoiClick(map)
        setMapStyle(map)
        enableMyLocation()
    }



    private fun openGoogleMapsForNavigation(destinationLatLng: LatLng) {
        val gmmIntentUri = Uri.parse("google.navigation:q=${destinationLatLng.latitude},${destinationLatLng.longitude}&mode=w")
        val mapIntent = Intent(Intent.ACTION_VIEW, gmmIntentUri)
        mapIntent.setPackage("com.google.android.apps.maps")

        if (mapIntent.resolveActivity(packageManager) != null) {
            startActivity(mapIntent)
        } else {
            Toast.makeText(this, "Google Maps is not installed", Toast.LENGTH_SHORT).show()
        }
    }


    private fun enableMyLocation() {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            return ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION),
                REQUEST_LOCATION_PERMISSION
            )
        }
        map.isMyLocationEnabled = true
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_LOCATION_PERMISSION) {
            if (grantResults.isNotEmpty() && (grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                enableMyLocation()
            }
        }
    }

    private fun setMapStyle(map: GoogleMap) {
        try {
            val success = map.setMapStyle(
                MapStyleOptions.loadRawResourceStyle(
                    this,
                    R.raw.map_style
                )
            )
            if (!success) {
                Log.e(TAG, "Style parsing failed.")
            }
        } catch (e: Resources.NotFoundException) {
            Log.e(TAG, "Can't find style. Error: ", e)
        }
    }

    private fun setPoiClick(map: GoogleMap) {
        map.setOnPoiClickListener { poi ->
            val poiMarker = map.addMarker(
                MarkerOptions()
                    .position(poi.latLng)
                    .title(poi.name)
            )
            poiMarker?.showInfoWindow()
        }
    }

    private fun setMapLongClick(map: GoogleMap) {
        map.setOnMapLongClickListener { latLng ->
            val snippet = String.format(
                Locale.getDefault(),
                "Lat: %1$.5f, Long: %2$.5f",
                latLng.latitude,
                latLng.longitude
            )
            map.addMarker(
                MarkerOptions()
                    .position(latLng)
                    .title(getString(R.string.dropped_pin))
                    .snippet(snippet)
                    .icon(BitmapDescriptorFactory.defaultMarker(BitmapDescriptorFactory.HUE_BLUE))
            )
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        textToSpeech.shutdown()
    }
}