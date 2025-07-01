"""FastHTML web interface for Voice-driven D&D framework"""

from pathlib import Path

from fasthtml.common import *
from starlette.responses import FileResponse

# Import your existing D&D framework components
from wip2 import (
    CampaignPlan,
    dm_turn,
    read_memory,
    STATE_SAVE_FILE,
    create_tts,
    DM_TTS_INSTRUCTIONS,
    AppState,
)
from stt import WhisperSTT
from image import generate_image

# Global state
app_state = {
    'plan': None,
    'stt_model': None,
    'dm_tts': None,
    'current_scene_image': None,
    'current_npc': None,
    'is_recording': False,
    'last_dm_response': '',
    'conversation_log': [],
}

# Initialize FastHTML app
app, rt = fast_app(
    hdrs=(
        # Add Web Speech API support
        Script("""
        // Global audio recording state
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        
        // Check for Web Speech API support
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        let recognition;
        
        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
        }
        
        function startVoiceInput() {
            if (!recognition) {
                alert('Speech recognition not supported in this browser');
                return;
            }
            
            const button = document.getElementById('voice-button');
            const status = document.getElementById('voice-status');
            
            if (isRecording) {
                recognition.stop();
                return;
            }
            
            isRecording = true;
            button.textContent = '🛑 Stop Recording';
            button.className = 'btn btn-danger';
            status.textContent = 'Listening... Speak now!';
            status.className = 'text-success';
            
            recognition.start();
            
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                document.getElementById('player-input').value = transcript;
                status.textContent = `Heard: "${transcript}"`;
                status.className = 'text-info';
                
                // Auto-submit the form
                document.getElementById('action-form').submit();
            };
            
            recognition.onerror = function(event) {
                status.textContent = `Error: ${event.error}`;
                status.className = 'text-danger';
                resetVoiceButton();
            };
            
            recognition.onend = function() {
                resetVoiceButton();
            };
        }
        
        function resetVoiceButton() {
            isRecording = false;
            const button = document.getElementById('voice-button');
            const status = document.getElementById('voice-status');
            button.textContent = '🎤 Start Voice Input';
            button.className = 'btn btn-primary';
            if (status.textContent.includes('Listening')) {
                status.textContent = 'Ready for voice input';
                status.className = 'text-muted';
            }
        }
        
        // Auto-play audio responses
        function playAudio(audioData) {
            if (audioData) {
                const audio = new Audio('data:audio/wav;base64,' + audioData);
                audio.play().catch(e => console.log('Audio play failed:', e));
            }
        }
        """),
        # Bootstrap CSS for styling
        Link(rel='stylesheet', href='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'),
        Style("""
        .scene-image {
            max-width: 100%;
            max-height: 400px;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .dm-response {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .player-action {
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #007bff;
            margin: 10px 0;
        }
        .voice-controls {
            background: #e9ecef;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .game-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        """),
    )
)


def initialize_game():
    """Initialize the game state"""
    if not app_state['plan']:
        if STATE_SAVE_FILE.exists():
            app_state['plan'] = CampaignPlan.model_validate_json(STATE_SAVE_FILE.read_text())
        else:
            # Redirect to campaign creation if no plan exists
            return False

    if not app_state['stt_model']:
        app_state['stt_model'] = WhisperSTT(model_name='base')

    if not app_state['dm_tts']:
        app_state['dm_tts'] = create_tts(voice_id='ash', instructions=DM_TTS_INSTRUCTIONS)

    return True


@rt('/')
def get():
    """Main game interface"""
    if not initialize_game():
        return Redirect('/setup')

    plan = app_state['plan']

    return Titled(
        Div(
            # Header
            Div(
                H1(f'🎲 {plan.title}', cls='text-center mb-3'),
                P(plan.synopsis, cls='lead text-center text-muted mb-4'),
                cls='mb-4',
            ),
            # Current scene image
            Div(
                H3('🖼️ Current Scene'),
                Div(
                    Img(
                        src=f'/image/{app_state["current_scene_image"]}'
                        if app_state['current_scene_image']
                        else '/static/placeholder.jpg',
                        alt='Current scene',
                        cls='scene-image',
                    )
                    if app_state['current_scene_image']
                    else P('No scene image yet', cls='text-muted'),
                    cls='text-center mb-4',
                ),
                cls='mb-4',
            ),
            # Voice input controls
            Div(
                H3('🎤 Voice Input'),
                Div(
                    Button(
                        '🎤 Start Voice Input',
                        id='voice-button',
                        cls='btn btn-primary btn-lg me-3',
                        onclick='startVoiceInput()',
                    ),
                    Span('Ready for voice input', id='voice-status', cls='text-muted'),
                    cls='mb-3',
                ),
                cls='voice-controls',
            ),
            # Manual input form
            Form(
                Div(
                    Label('Or type your action:', cls='form-label'),
                    Textarea(
                        placeholder='What do you do?',
                        name='player_action',
                        id='player-input',
                        cls='form-control mb-3',
                        rows='3',
                    ),
                    Button('Submit Action', type='submit', cls='btn btn-success'),
                    cls='mb-4',
                ),
                method='post',
                action='/action',
                id='action-form',
                # TODO disable form until input is ready, then disable again, while awaiting the response
            ),
            # Game log
            Div(
                H3('📜 Adventure Log'),
                Div(
                    *[
                        Div(
                            Div(Strong('You: '), entry['player_action'], cls='player-action'),
                            Div(Strong('DM: '), entry['dm_response'], cls='dm-response'),
                        )
                        for entry in reversed(app_state['conversation_log'][-5:])  # Show last 5 exchanges
                    ]
                    if app_state['conversation_log']
                    else [P('Your adventure begins here...', cls='text-muted')],
                    id='game-log',
                ),
                cls='mb-4',
            ),
            # Memory/Notes section
            Details(Summary('📝 Campaign Notes'), Pre(read_memory(), cls='bg-light p-3 rounded'), cls='mb-4'),
            cls='game-container',
        ),
    )


@rt('/action', methods=['POST'])
def post_action(player_action: str):
    """Handle player actions"""
    if not initialize_game():
        return Redirect('/setup')

    if not player_action.strip():
        return Redirect('/')

    plan = app_state['plan']

    # Get DM response
    dm_response = dm_turn(player_action, plan)

    # Generate scene image
    scene_prompt = f'{plan.visual_style}, {dm_response.scene_description}'
    negative_prompt = 'blurry, low quality, distorted, modern elements, text, watermark, UI elements'

    scene_images = generate_image(scene_prompt, negative_prompt, 1)
    app_state['current_scene_image'] = scene_images[0].name

    # Update conversation log
    app_state['conversation_log'].append({'player_action': player_action, 'dm_response': dm_response.gm_speech})

    # Store last DM response for audio playback
    app_state['last_dm_response'] = dm_response.gm_speech

    # Check if NPC conversation should start
    if dm_response.npc:
        app_state['current_npc'] = dm_response.npc
        return Redirect('/npc')

    return Redirect('/')


@rt('/npc')
def get_npc():
    """NPC conversation interface"""
    if not app_state['current_npc']:
        return Redirect('/')

    npc = app_state['current_npc']
    plan = app_state['plan']

    # Generate NPC portrait
    npc_prompt = f'{plan.visual_style}, character portrait, {npc.visual_description}'
    negative_prompt = 'blurry, low quality, distorted, multiple people, crowd, background characters, text, watermark'
    npc_images = generate_image(npc_prompt, negative_prompt, 1)
    npc_image = npc_images[0].name

    return Titled(
        Div(
            H1(f'💬 Conversation with {npc.name}', cls='text-center mb-4'),
            # NPC Portrait
            Div(
                Img(src=f'/image/{npc_image}', alt=f'{npc.name}', cls='scene-image mb-3'),
                P(npc.role, cls='text-muted'),
                cls='text-center mb-4',
            ),
            # Voice input for NPC conversation
            Div(
                Button(
                    '🎤 Speak to NPC', id='voice-button', cls='btn btn-primary btn-lg me-3', onclick='startVoiceInput()'
                ),
                Span('Ready to talk', id='voice-status', cls='text-muted'),
                cls='voice-controls mb-4',
            ),
            # Manual input
            Form(
                Textarea(
                    placeholder=f'What do you say to {npc.name}?',
                    name='npc_input',
                    id='player-input',
                    cls='form-control mb-3',
                    rows='3',
                ),
                Button('Say It', type='submit', cls='btn btn-success me-2'),
                Button('End Conversation', formaction='/end_npc', cls='btn btn-secondary'),
                method='post',
                action='/npc_talk',
                id='action-form',
            ),
            cls='game-container',
        ),
    )


@rt('/npc_talk', methods=['POST'])
def post_npc_talk(npc_input: str):
    """Handle NPC conversation input"""
    # This would integrate with your existing npc_loop logic
    # For now, redirect back to main game
    app_state['current_npc'] = None
    return Redirect('/')


@rt('/end_npc', methods=['POST'])
def post_end_npc():
    """End NPC conversation"""
    app_state['current_npc'] = None
    return Redirect('/')


@rt('/image/{filename}')
def get_image(filename: str):
    print('Calling get_image')
    print(filename)
    """Serve generated images"""
    image_path = Path('cache/images') / filename
    print(image_path)
    if image_path.exists():
        return FileResponse(image_path)
    return Response('Image not found', status_code=404)


@rt('/setup')
def get_setup():
    """Campaign setup page"""
    return Titled(
        '🎲 D&D Campaign Setup',
        Div(
            H1('🎲 Create Your D&D Campaign', cls='text-center mb-4'),
            P('No campaign found. Please create one using the command line interface first.', cls='text-center'),
            A('Go back to main game', href='/', cls='btn btn-primary'),
            cls='game-container text-center',
        ),
    )


if __name__ == '__main__':
    serve()
