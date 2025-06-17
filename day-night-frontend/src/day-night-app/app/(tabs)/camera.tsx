import React, { useEffect, useRef, useState } from 'react';
import { Button, View, Text, StyleSheet, Image } from 'react-native';
import { Camera } from 'expo-camera';
import type { Camera as CameraComponentType } from 'expo-camera';
import * as MediaLibrary from 'expo-media-library';

export default function CameraScreen() {
    const [hasPermission, setHasPermission] = useState<boolean | null>(null);
    const [photo, setPhoto] = useState<string | null>(null);
    const cameraRef = useRef<typeof Camera>(null);



    useEffect(() => {
        (async () => {
            const { status } = await Camera.requestCameraPermissionsAsync();
            await MediaLibrary.requestPermissionsAsync();
            setHasPermission(status === 'granted');
        })();
    }, []);

    const takePhoto = async () => {
        if (cameraRef.current) {
            const data = await cameraRef.current.takePictureAsync();
            setPhoto(data.uri);
        }
    };

    const handleCheck = async () => {
        if (!photo) return;

        const formData = new FormData();
        formData.append('file', {
            uri: photo,
            name: 'photo.jpg',
            type: 'image/jpeg',
        } as any);

        try {
            const res = await fetch('http://YOUR_BACKEND_URL/predict', {
                method: 'POST',
                body: formData,
            });

            const data = await res.json();
            alert(`It's a ${data.result}`);
        } catch (err) {
            console.error(err);
            alert('Failed to classify the image.');
        }
    };

    if (hasPermission === null) return <Text>Requesting permission...</Text>;
    if (hasPermission === false) return <Text>No access to camera</Text>;

    return (
        <View style={styles.container}>
            {photo ? (
                <>
                    <Image source={{ uri: photo }} style={styles.preview} />
                    <Button title="Check" onPress={handleCheck} />
                    <Button title="Retake" onPress={() => setPhoto(null)} />
                </>
            ) : (
                <>
                    <Camera style={styles.camera} ref={cameraRef} />
                    <Button title="Take Photo" onPress={takePhoto} />
                </>
            )}
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1 },
    camera: { flex: 1 },
    preview: { flex: 1 },
});
