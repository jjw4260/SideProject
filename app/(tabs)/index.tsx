import { Audio } from 'expo-av';
import * as ImagePicker from 'expo-image-picker';
import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  Modal,
  Platform,
  SafeAreaView,
  ScrollView,
  Text,
  TextInput,
  TouchableOpacity,
  View
} from 'react-native';
import styled from 'styled-components/native';

// Types
interface PredictionResult {
  title: string;
  artist: string;
  cover_url: string;
  lyrics: string;
}

// 서버 주소
const HOST = Platform.select({ ios: 'localhost', android: '10.0.2.2', default: '192.0.0.2' });
const SERVER_URL = `http://${HOST}:5001`;

// Styled Components
const Container = styled(SafeAreaView)`
  flex: 1;
  background-color: #667eea;
  padding-bottom: 160px;
`;
const HeaderRow = styled.View`
  height: 60px;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
`;
const IconButton = styled(TouchableOpacity)`
  padding: 8px;
`;
const HeaderTitle = styled.Text`
  color: #fff;
  font-size: 20px;
  font-weight: bold;
`;
const AlbumContainer = styled.View`
  flex: 1;
  align-items: center;
  justify-content: center;
`;
const Placeholder = styled.View`
  width: 160px;
  height: 160px;
  background-color: rgba(255,255,255,0.3);
  border-radius: 12px;
`;
const AlbumCover = styled(Image)`
  width: 160px;
  height: 160px;
  border-radius: 12px;
`;
const SongTitle = styled.Text`
  margin-top: 16px;
  color: #fff;
  font-size: 22px;
  font-weight: bold;
`;
const Artist = styled.Text`
  margin-top: 4px;
  color: #e0e0e0;
  font-size: 18px;
`;
const InputRow = styled.View`
  flex-direction: row;
  align-items: center;
  padding: 0 16px 60px;
`;
const SceneInput = styled(TextInput)`
  flex: 1;
  height: 48px;
  background-color: #fff;
  border-radius: 24px;
  padding: 0 16px;
  font-size: 16px;
`;

export default function App() {
  const [sceneText, setSceneText] = useState('');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [lyricsVisible, setLyricsVisible] = useState(false);
  const [lyrics, setLyrics] = useState('');

  useEffect(() => {
    (async () => {
      const { status: micStatus } = await Audio.requestPermissionsAsync();
      if (micStatus !== 'granted') Alert.alert('권한 오류', '마이크 권한이 필요합니다');
      const { status: imgStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (imgStatus !== 'granted') Alert.alert('권한 오류', '갤러리 권한이 필요합니다');
    })();
  }, []);

  // 텍스트 예측
  const handleTextPredict = async () => {
    if (!sceneText.trim()) {
      Alert.alert('알림', '장면 설명을 입력하세요');
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`${SERVER_URL}/predict/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: sceneText }),
      });
      if (!res.ok) throw new Error(`서버 오류 ${res.status}`);
      const json = await res.json();
      setResult(json);
      setLyrics(json.lyrics || '');
    } catch (e: any) {
      Alert.alert('오류', e.message);
    } finally {
      setLoading(false);
    }
  };

  // 이미지 예측
  const handleImagePredict = async () => {
    try {
      const picker = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        quality: 0.7,
      });
      if (picker.canceled) return;
      const { uri } = picker.assets[0];
      setLoading(true);

      const form = new FormData();
      form.append('file', {
        uri,
        name: uri.split('/').pop(),
        type: 'image/jpeg',
      } as any);

      const res = await fetch(`${SERVER_URL}/predict_image`, {
        method: 'POST',
        body: form,
      });
      if (!res.ok) throw new Error(`서버 오류 ${res.status}`);
      const json = await res.json();
      setResult(json);
      setLyrics(json.lyrics || '');
    } catch (e: any) {
      Alert.alert('이미지 오류', e.message);
    } finally {
      setLoading(false);
    }
  };

  // 음악 인식: 10초 자동 녹음 후 예측
  const handleAudioPredict = async () => {
    setLoading(true);
    try {
      await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      setRecording(recording);

      setTimeout(async () => {
        try {
          await recording.stopAndUnloadAsync();
          const uri = recording.getURI();
          if (uri) {
            const form = new FormData();
            form.append('file', { uri, name: 'audio.wav', type: 'audio/wav' } as any);
            const res = await fetch(`${SERVER_URL}/predict_audio`, { method: 'POST', body: form });
            if (!res.ok) throw new Error(`서버 오류 ${res.status}`);
            const json = await res.json();
            setResult(json);
            setLyrics(json.lyrics || '');
          }
        } catch (innerErr: any) {
          Alert.alert('음악 인식 오류', innerErr.message);
        } finally {
          setLoading(false);
        }
      }, 10000);
    } catch (e: any) {
      Alert.alert('음악 인식 오류', e.message);
      setLoading(false);
    }
  };

  // 수동 오디오 예측 (uri 전달)
  const predictAudio = async (uri: string) => {
    setLoading(true);
    try {
      const form = new FormData();
      form.append('file', { uri, name: 'audio.wav', type: 'audio/wav' } as any);
      const res = await fetch(`${SERVER_URL}/predict_audio`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`서버 오류 ${res.status}`);
      const json = await res.json();
      setResult(json);
      setLyrics(json.lyrics || '');
    } catch (e: any) {
      Alert.alert('오류', e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container>
      <HeaderRow>
        <IconButton onPress={handleAudioPredict}>
          <Image
            source={require('/Users/jungwoong-jeon/Documents/Project/predict_ost_app/app/(tabs)/assets/images/mic2.png')}
            style={{ width: 24, height: 24, tintColor: '#fff' }}
          />
        </IconButton>
        <HeaderTitle>DeepTune</HeaderTitle>
        <IconButton onPress={handleImagePredict}>
          <Image
            source={require('/Users/jungwoong-jeon/Documents/Project/predict_ost_app/app/(tabs)/assets/images/photo.png')}
            style={{ width: 24, height: 24, tintColor: '#fff' }}
          />
        </IconButton>
      </HeaderRow>

      <AlbumContainer>
        {result?.cover_url ? (
          <AlbumCover source={{ uri: result.cover_url }} />
        ) : (
          <Placeholder />
        )}
        <SongTitle>{result?.title || '노래 제목'}</SongTitle>
        <Artist>{result?.artist || '아티스트'}</Artist>
      </AlbumContainer>

      {result?.lyrics ? (
        <TouchableOpacity
          style={{
            backgroundColor: 'rgba(255,255,255,0.3)',
            paddingVertical: 10,
            paddingHorizontal: 20,
            borderRadius: 20,
            alignSelf: 'center',
            marginBottom: 16
          }}
          onPress={() => setLyricsVisible(true)}
        >
          <Text style={{ color: '#fff', fontSize: 16 }}> 가사 보기</Text>
        </TouchableOpacity>
      ) : null}

      <InputRow>
        <SceneInput
          placeholder="장면 설명을 입력하세요"
          value={sceneText}
          onChangeText={setSceneText}
          returnKeyType="send"
          onSubmitEditing={handleTextPredict}
        />
        <IconButton onPress={handleTextPredict}>
          <Text style={{ color: '#fff', fontSize: 16, fontWeight: '600', marginLeft: 8 }}>
            Click
          </Text>
        </IconButton>
      </InputRow>

   <Modal
  visible={lyricsVisible}
  animationType="slide"
  transparent
  onRequestClose={() => setLyricsVisible(false)}
>
  <TouchableOpacity
    activeOpacity={1}
    style={{
      flex: 1,
      backgroundColor: 'rgba(0,0,0,0.75)',
      justifyContent: 'flex-end'
    }}
    onPress={() => setLyricsVisible(false)}
  >
    <View
      style={{
        maxHeight: '70%',
        minHeight: 180,
        backgroundColor: '#1e1e1ecc',
        borderTopLeftRadius: 24,
        borderTopRightRadius: 24,
        padding: 20
      }}
    >
      <ScrollView showsVerticalScrollIndicator={true}>
        <Text
          style={{
            color: '#fff',
            fontSize: 16,
            lineHeight: 24,
            textAlign: 'center',
            paddingBottom: 12
          }}
        >
          {lyrics}
        </Text>
      </ScrollView>
    </View>
  </TouchableOpacity>
</Modal>

      {loading && (
        <ActivityIndicator
          size="large"
          color="#fff"
          style={{ position: 'absolute', top: '50%', left: '50%' }}
        />
      )}
    </Container>
  );
}
