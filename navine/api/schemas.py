from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class TextResponse(BaseModel):
    text: str


class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    num_steps: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    enhance: bool = True
    model_profile: Optional[str] = None

    def resolved_steps(self) -> Optional[int]:
        if self.num_steps is not None:
            return self.num_steps
        return self.steps


class ImageResponse(BaseModel):
    path: str
    mime: str
    data_base64: str
    enhanced_prompt: Optional[str] = None
    reference_used: bool = False
    keyword_matched: Optional[str] = None
    reference_category: Optional[str] = None
    reference_strength: Optional[float] = None


class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    num_frames: Optional[int] = None
    fps: Optional[int] = None
    seed: Optional[int] = None
    audio: bool = False
    narration: Optional[str] = None
    voice: Optional[str] = None
    model_profile: Optional[str] = None


class DeepfakeFaceRequest(BaseModel):
    source: Optional[str] = None
    target: Optional[str] = None
    source_base64: Optional[str] = None
    target_base64: Optional[str] = None
    strength: float = 0.85


class DeepfakeVideoRequest(BaseModel):
    source: str = Field(..., min_length=1)
    target_video: Optional[str] = None
    prompt: Optional[str] = None
    num_frames: int = 16
    fps: int = 12
    strength: float = 0.85
    seed: Optional[int] = None
    audio: bool = False
    narration: Optional[str] = None


class OsintInvestigateRequest(BaseModel):
    target: str = Field(..., min_length=1)
    kind: str = "auto"
    use_search: bool = True
    output_dir: Optional[str] = None


class OsintInvestigateResponse(BaseModel):
    ok: bool = True
    target: str
    kind: str
    query: str
    analysis: str
    search_results: list[dict] = Field(default_factory=list)
    username_scan: Optional[dict] = None
    domain_intel: Optional[dict] = None
    ip_intel: Optional[dict] = None
    report_path: Optional[str] = None


class VoiceCloneRequest(BaseModel):
    name: str = Field(..., min_length=1)
    sample_path: Optional[str] = None
    sample_base64: Optional[str] = None
    language: str = "en"


class VoiceSpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: Optional[str] = None
    play: bool = False


class VoiceTranscribeRequest(BaseModel):
    audio_base64: str = Field(..., min_length=1)
    mime: str = "audio/wav"
    filename: Optional[str] = None


class MusicRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    duration: float = Field(default=12.0, ge=4.0, le=30.0)
    seed: Optional[int] = None


class MusicResponse(BaseModel):
    ok: bool = True
    path: str
    mime: str = "audio/wav"
    data_base64: Optional[str] = None
    bytes: int = 0
    meta: dict = Field(default_factory=dict)


class CameraDescribeRequest(BaseModel):
    image_base64: str = Field(..., min_length=1)
    mime: str = "image/jpeg"
    question: Optional[str] = None
    use_model: bool = True
    include_screen: bool = False


class ApiKeyCreateRequest(BaseModel):
    name: str = "Navine AI - Python"
    scopes: Optional[list[str]] = None
    expires_days: Optional[int] = None


class ApiKeyCreateResponse(BaseModel):
    id: str
    name: str
    key: str
    prefix: Optional[str] = None
    scopes: list[str] = Field(default_factory=list)
    expires_at: Optional[str] = None


class ApiKeyInfo(BaseModel):
    id: str
    name: str
    prefix: Optional[str] = None
    scopes: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    last_used_at: Optional[str] = None
    use_count: int = 0
    revoked: bool = False
    owner: Optional[str] = None


class ApiKeyListResponse(BaseModel):
    keys: list[ApiKeyInfo]


class VideoResponse(BaseModel):
    path: str
    mime: str
    data_base64: str
    reference_used: bool = False
    keyword_matched: Optional[str] = None
    reference_category: Optional[str] = None
    reference_strength: Optional[float] = None


class CodeRequest(BaseModel):
    task: str = Field(..., min_length=1)
    language: Optional[
        Literal["python", "javascript", "typescript", "rust", "go", "java", "cpp"]
    ] = None


class CodeResponse(BaseModel):
    code: str
    language: str


class CodeRunRequest(BaseModel):
    code: str = Field(..., min_length=1)
    language: str = Field(default="python", min_length=1, max_length=32)
    timeout: Optional[float] = Field(default=None, ge=1.0, le=60.0)


class CodeRunResponse(BaseModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    language: str = "python"


class HealthResponse(BaseModel):
    status: str


class ModelStatus(BaseModel):
    name: str
    trained: bool
    checkpoint: str
    mtime: Optional[str] = None
    age_seconds: Optional[float] = None
    training: bool = False


class ProductProgress(BaseModel):
    name: str
    phase: Optional[str] = None
    score: Optional[float] = None
    streak: Optional[int] = None
    sample: Optional[str] = None
    active: bool = False


class TrainProgressResponse(BaseModel):
    running: bool = False
    cycle: int = 0
    max_cycles: Optional[int] = None
    best_scores: dict = Field(default_factory=dict)
    consecutive_pass: dict = Field(default_factory=dict)
    last_line: Optional[str] = None
    recent_lines: list[str] = Field(default_factory=list)
    models: list[ModelStatus] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list)
    product: str = "Navine AI - Python"
    active_product: Optional[str] = None
    stage: Optional[str] = None
    stage_index: int = 0
    stage_total: int = 0
    steps: int = 0
    steps_total: int = 0
    label: Optional[str] = None
    percent: float = 0.0
    elapsed_s: Optional[int] = None
    eta_s: Optional[int] = None
    updated_at: Optional[str] = None
    products: list[ProductProgress] = Field(default_factory=list)
    locks: dict = Field(default_factory=dict)
    marathon_round: Optional[int] = None
    marathon_loop: bool = False


class TrainingTypeInfo(BaseModel):
    name: str
    description: str
    data_format: str


class InfoResponse(BaseModel):
    pytorch: str
    cuda_available: bool
    gpu: Optional[str] = None
    models: list[ModelStatus]
    local_only: bool = True
    unrestricted: bool = True
    product: str = "Navine AI - Python"
    engine: Optional[str] = None
    default_text_model: Optional[str] = None
    port: Optional[int] = None
    modes: Optional[list[str]] = None
    training_types: Optional[list[TrainingTypeInfo]] = None
    theme_id: Optional[str] = None
    theme: Optional[dict] = None
    public_url: Optional[str] = None
    tagline: Optional[str] = None
    backend: Optional[str] = None
    python_only: bool = False
    ui_tabs: Optional[list[str]] = None
    ui_features: Optional[dict] = None
    runtime: Optional[dict] = None


class ModelStatDetail(BaseModel):
    name: str
    label: str = ""
    modality: str = "text"
    category: str = "text"
    parameters: Optional[int] = None
    parameters_human: Optional[str] = None
    parameters_exact: Optional[str] = None
    parameter_source: Optional[str] = None
    architecture: dict = Field(default_factory=dict)
    architecture_source: Optional[str] = None
    trained: bool = False
    checkpoint: str = ""
    checkpoint_key: Optional[str] = None
    shares_checkpoint_with: Optional[str] = None
    mtime: Optional[str] = None
    age_seconds: Optional[float] = None
    training: bool = False
    config_name: Optional[str] = None


class InferenceProfileDetail(BaseModel):
    modality: str
    mode: str
    label: str = ""
    config: str = ""
    params: dict = Field(default_factory=dict)


class ModelsStatsResponse(BaseModel):
    product: str = "Navine AI - Python"
    port: Optional[int] = None
    default_text_model: Optional[str] = None
    updated_at: Optional[str] = None
    total_parameters: Optional[int] = None
    total_parameters_human: Optional[str] = None
    total_parameters_exact: Optional[str] = None
    parameter_source: Optional[str] = None
    trained_models: int = 0
    models: list[ModelStatDetail] = Field(default_factory=list)
    inference_profiles: list[InferenceProfileDetail] = Field(default_factory=list)
    host_specs: Optional[dict] = None
    capabilities: list[str] = Field(default_factory=list)


class TrainingTypesResponse(BaseModel):
    product: str = "Navine AI - Python"
    types: list[TrainingTypeInfo]


class AutolearnStatusResponse(BaseModel):
    enabled: bool
    sources: list[str]
    last_run: Optional[str] = None
    last_fetch_count: int = 0
    last_new_samples: int = 0
    total_samples: int = 0
    samples_since_train: int = 0
    next_train_in: int = 0
    last_train: Optional[str] = None
    last_by_category: dict = Field(default_factory=dict)
    auto_train: bool = True
    interval_minutes: int = 60
    product: str = "Navine AI - Python"


class MarathonStatusResponse(BaseModel):
    running: bool = False
    started_at: Optional[str] = None
    target_hours: Optional[float] = None
    forever: bool = False
    elapsed_hours: float = 0.0
    eta_hours: Optional[float] = None
    cycle_count: int = 0
    items_fetched: int = 0
    items_ingested: int = 0
    images_fetched: int = 0
    videos_fetched: int = 0
    train_count: int = 0
    deep_train_count: int = 0
    last_train: Optional[str] = None
    last_deep_train: Optional[str] = None
    last_cycle_at: Optional[str] = None
    current_source: Optional[str] = None
    samples_since_train: int = 0
    stopped_reason: Optional[str] = None
    last_error: Optional[str] = None
    pid: Optional[int] = None
    product: str = "Navine AI - Python"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1)


class ChatTurn(BaseModel):
    user: str
    assistant: str


class ChatRequest(BaseModel):
    messages: Optional[list[ChatMessage]] = None
    message: Optional[str] = None
    history: list[ChatTurn] = Field(default_factory=list)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    use_rag: bool = True
    use_search: bool = False
    session_id: Optional[str] = None
    model_profile: Optional[str] = None
    allow_emoji: bool = True
    image_base64: Optional[str] = None
    image_mime: Optional[str] = None


class UserSettings(BaseModel):
    allow_emoji: bool = True
    discord_webhook_url: Optional[str] = ""


class UserSettingsResponse(BaseModel):
    allow_emoji: bool = True
    discord_webhook_url: Optional[str] = ""
    discord_webhook_configured: bool = False



class ChatResponse(BaseModel):
    text: str
    message: Optional[ChatMessage] = None
    is_code: bool = False
    searched: bool = False
    search_attempted: bool = False
    session_id: Optional[str] = None
    model_profile: Optional[str] = None


class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[OpenAIChatMessage] = Field(..., min_length=1)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False


class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIChatMessage
    finish_reason: str


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChatChoice]
    usage: OpenAIUsage


class DesktopStatusResponse(BaseModel):
    ok: bool = True
    allow_screen_config: bool = True
    allow_input_config: bool = False
    screen_enabled: bool = True
    input_enabled: bool = False
    require_explicit_request: bool = True
    max_type_chars: int = 2000
    output_dir: Optional[str] = None
    warning: Optional[str] = None
    session: Optional[dict] = None


class DesktopSettingsRequest(BaseModel):
    allow_screen: Optional[bool] = None
    allow_input: Optional[bool] = None


class ScreenCaptureRequest(BaseModel):
    path: Optional[str] = None


class ScreenDescribeRequest(BaseModel):
    question: Optional[str] = None
    use_model: bool = True


class DesktopInputRequest(BaseModel):
    action: Literal["type", "click", "hotkey"]
    text: Optional[str] = None
    keys: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    button: str = "left"
    clicks: int = 1


class FileCreateRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=180)
    content: str = Field(..., min_length=0)
    mime: Optional[str] = None


class FileCreateResponse(BaseModel):
    ok: bool = True
    path: str
    filename: str
    mime: str
    size: int
    download_url: str
    data_base64: Optional[str] = None


class ZipFileEntry(BaseModel):
    path: str = Field(..., min_length=1, max_length=180)
    content: str = ""


class ZipCreateRequest(BaseModel):
    filename: str = Field(default="archive.zip", min_length=1, max_length=180)
    files: List[ZipFileEntry] = Field(default_factory=list, min_length=1)


class McpServerUpsertRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=48)
    command: Optional[str] = ""
    args: List[str] = Field(default_factory=list)
    url: Optional[str] = ""
    enabled: bool = True
    cwd: Optional[str] = ""


class McpServerRemoveResponse(BaseModel):
    ok: bool = True
    removed: str


class AdminUsernameRequest(BaseModel):
    username: str = Field(..., min_length=1)


class AdminLoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class AdminLoginResponse(BaseModel):
    ok: bool = True
    token: str
    username: str
    expires_at: int


class AdminSessionResponse(BaseModel):
    ok: bool = True
    username: str
    expires_at: Optional[int] = None
    is_admin: bool = False


class AuthSignupRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=32)
    password: str = Field(..., min_length=4)


class AuthLoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class AuthLoginResponse(BaseModel):
    ok: bool = True
    token: str
    username: str
    expires_at: int
    is_admin: bool = False


class AuthSessionResponse(BaseModel):
    ok: bool = True
    username: str
    expires_at: Optional[int] = None
    is_admin: bool = False


class UserChatSummary(BaseModel):
    id: str
    title: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    message_count: int = 0


class UserChatListResponse(BaseModel):
    ok: bool = True
    chats: list[UserChatSummary] = Field(default_factory=list)


class UserChatCreateRequest(BaseModel):
    title: Optional[str] = None


class UserChatRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=80)


class UserChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class UserChatDetailResponse(BaseModel):
    ok: bool = True
    id: str
    title: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    messages: list[UserChatMessage] = Field(default_factory=list)


class TrainStartRequest(BaseModel):
    target: str = Field(..., min_length=1)
    steps: Optional[int] = None
    config: Optional[str] = None
    language: Optional[str] = None
    require_cuda: bool = True
    voice_name: Optional[str] = None
    voice_sample: Optional[str] = None
    voice_sample_base64: Optional[str] = None
    voice_set_default: bool = True
    deepfake_strength: Optional[float] = None
    deepfake_source: Optional[str] = None
    deepfake_target: Optional[str] = None
    deepfake_source_base64: Optional[str] = None
    deepfake_target_base64: Optional[str] = None


class TrainCustomRequest(BaseModel):
    source: str = Field(..., min_length=1)
    url: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    train_target: str = "text"
    steps: Optional[int] = 400
    config: Optional[str] = None
    language: Optional[str] = None
    require_cuda: bool = True
    voice_name: Optional[str] = None
    voice_sample: Optional[str] = None
    voice_sample_base64: Optional[str] = None
    voice_set_default: bool = True
    deepfake_strength: Optional[float] = None
    deepfake_source: Optional[str] = None
    deepfake_target: Optional[str] = None
    deepfake_source_base64: Optional[str] = None
    deepfake_target_base64: Optional[str] = None


class TrainJobResponse(BaseModel):
    ok: bool = True
    started: bool = True
    job: dict = Field(default_factory=dict)


class TrainJobsResponse(BaseModel):
    ok: bool = True
    jobs: list[dict] = Field(default_factory=list)


class TrainLearnRequest(BaseModel):
    action: str = Field(..., min_length=1)
    url: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    max_pages: Optional[int] = 8

