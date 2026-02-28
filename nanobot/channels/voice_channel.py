"""Baidu Voice Channel Implementation for nanobot"""
import asyncio
import pyaudio
from aip import AipSpeech
from typing import Optional, Dict, Any
from nanobot.channels.base import BaseChannel
from nanobot.config import Config
from nanobot.bus.queue import MessageBus
from loguru import logger

class VoiceChannel(BaseChannel):
    """Voice channel using Baidu Speech API"""
    name = "voice"

    def __init__(self, config: Config, bus: MessageBus):
        super().__init__(config.channels.voice, bus)
        self.voice_config = config.channels.voice
        self.client = AipSpeech(
            self.voice_config.app_id,
            self.voice_config.baidu_api_key,
            self.voice_config.baidu_secret_key
        )
        # 音频采集配置
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # 百度API要求16kHz采样率
        self.chunk = 1024
        self.record_seconds = 3  # 单次录音时长
        self.p = pyaudio.PyAudio()
        self.stream = None
        self._recording_task = None

        logger.info(f"Baidu voice channel initialized with provider: {self.voice_config.provider}")

    async def start(self):
        """Start voice channel service with audio listening"""
        self._running = True
        # 启动音频监听任务
        self._recording_task = asyncio.create_task(self._listen_for_audio())
        logger.info("Voice channel started successfully (listening for audio input)")

    async def stop(self):
        """Stop voice channel service"""
        self._running = False
        if self._recording_task:
            self._recording_task.cancel()
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.warning(f"Error closing audio stream: {str(e)}")
        self.p.terminate()
        logger.info("Voice channel stopped")

    async def send(self, msg: Any) -> bool:
        """Send text message via voice channel (synthesize speech)"""
        try:
            text = msg.content if hasattr(msg, 'content') else str(msg)
            audio_data = await self.synthesize_speech(text)

            if audio_data:
                logger.info(f"Generated speech audio of {len(audio_data)} bytes")
                await self._play_audio(audio_data)  # 自动播放合成的语音
                return True
            else:
                logger.error("Failed to synthesize speech audio")
                return False

        except Exception as e:
            logger.error(f"Failed to send voice message: {str(e)}")
            return False

    async def _listen_for_audio(self):
        """Listen for audio input with voice activity detection"""
        while self._running:
            try:
                self.stream = self.p.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    frames_per_buffer=self.chunk
                )

                logger.debug("Listening for audio input...")
                frames = []
                has_voice = False

                # 录制音频
                for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                    try:
                        data = self.stream.read(self.chunk)
                        frames.append(data)

                        # 简单的语音激活检测：检查音频是否超过阈值
                        audio_intensity = max(data)
                        if audio_intensity > 100:  # 阈值可以根据环境调整
                            has_voice = True

                    except IOError as e:
                        if e.args[0] == -9981:
                            logger.warning("Audio input overflow, skipping frame")
                            continue
                        else:
                            raise e

                if has_voice:
                    audio_data = b''.join(frames)
                    logger.info(f"Recorded audio: {len(audio_data)} bytes")

                    # 异步识别语音
                    text = await self.recognize_speech(audio_data)
                    if text:
                        logger.info(f"Recognized speech: {text}")
                        # 将识别结果发布为InboundMessage
                        await self._handle_message(
                            sender_id="voice_user",
                            chat_id="voice_chat",
                            content=text
                        )
                    else:
                        logger.warning("No speech recognized")
                else:
                    logger.debug("No voice detected, skipping")

                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio listening error: {str(e)}")
                if self.stream is not None:
                    try:
                        self.stream.stop_stream()
                        self.stream.close()
                    except:
                        pass
                    await asyncio.sleep(1)

    async def recognize_speech(self, audio_data: bytes) -> Optional[str]:
        """Convert audio to text using Baidu ASR"""
        try:
            result = self.client.asr(
                audio_data,
                'pcm',
                16000,
                {
                    'dev_pid': 1537,  # 普通话识别
                }
            )

            if result.get('err_no') == 0:
                return result.get('result')[0]
            else:
                logger.error(f"Baidu ASR error: {result.get('err_msg')}")
        except Exception as e:
            logger.error(f"Baidu ASR request failed: {str(e)}")

        return None

    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Convert text to audio with correct parameter handling"""
        try:
            # 严格按照百度API要求设置参数
            params = {
                'vol': max(0, min(15, getattr(self.voice_config, 'vol', 5))),  # 0-15
                'spd': max(0, min(9, int(getattr(self.voice_config, 'tts_speed', 0.8) * 5))),  # 0-9
                'pit': max(0, min(9, getattr(self.voice_config, 'pit', 5))),  # 0-9
                'per': getattr(self.voice_config, 'tts_voice_name', 4)  # 1-11
            }

            # 使用百度API支持的语言代码
            language = getattr(self.voice_config, 'language', 'zh')
            if language not in ['zh', 'en']:
                language = 'zh'  # 默认中文

            result = self.client.synthesis(
                text,
                language,
                1,
                params
            )

            if not isinstance(result, dict):
                return result
            else:
                logger.error(f"Baidu TTS error: {result.get('err_msg')}")
                logger.debug(f"TTS params: {params}, text length: {len(text)}")  # 调试日志

        except Exception as e:
            logger.error(f"Baidu TTS request failed: {str(e)}")

        return None

    async def _play_audio(self, audio_data: bytes) -> None:
        """Play audio data using pyaudio"""
        try:
            # 直接使用pyaudio播放，避免依赖pygame
            stream = self.p.open(
                format=self.p.get_format_from_width(2),  # 16位音频
                channels=1,
                rate=16000,
                output=True
            )
            stream.write(audio_data)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            logger.error(f"Failed to play audio: {str(e)}")