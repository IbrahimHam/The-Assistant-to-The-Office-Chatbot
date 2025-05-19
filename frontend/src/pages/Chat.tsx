import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { Character } from '@/types';
import CharacterSprite from '@/components/CharacterSprite';
import ChatBubble from '@/components/ChatBubble';
import AchievementPopup from '@/components/AchievementPopup';
import useChat from '@/hooks/useChat';

const Chat = () => {
  const {
    messages,
    loading,
    currentCharacter,
    dundiesPoints,
    userName,
    sendMessage,
    switchCharacter,
  } = useChat({ character: 'Pam' });

  const [input, setInput] = useState('');
  const [showRoster, setShowRoster] = useState(false);
  const [achievements, setAchievements] = useState<any[]>([]);

  const messagesEndRef = useRef<null | HTMLDivElement>(null);
  const inputRef = useRef<null | HTMLInputElement>(null);

  const characters: Character[] = [
    'Michael',
    'Jim',
    'Pam',
    'Dwight',
    'Angela',
    'Oscar',
    'Kevin',
    'Stanley',
    'Creed',
    'Toby',
  ];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      sendMessage(input);
      setInput('');
    }
  };

  const toggleRoster = () => {
    setShowRoster(!showRoster);
    const toggleSound = new Audio(
      'https://assets.mixkit.co/sfx/preview/mixkit-unlock-game-notification-253.mp3'
    );
    toggleSound.volume = 0.2;
    toggleSound.play();
  };

  const handleCharacterSelect = (character: Character) => {
    switchCharacter(character);
    setShowRoster(false);
  };

  const removeAchievement = (id: string) => {
    setAchievements(achievements.filter((a) => a.id !== id));
  };

  return (
    <div className="relative min-h-screen bg-officeBackground flex flex-col">
      {/* Header */}
      <div className="bg-dundies p-4 pixel-border fixed top-0 left-0 w-full z-50">
        <div className="flex justify-between items-center">
          <Link to="/" className="font-pixel text-black text-xl flex items-center">
            <img
              src="https://i.ibb.co/w7QW1yk/worlds-best-boss-mug.png"
              alt="The Office Chatbot"
              className="w-8 h-8 mr-2"
            />
            <span className="hidden sm:inline">Assistant to The Office Chatbot</span>
            <span className="sm:hidden">The Office</span>
          </Link>

          <div className="flex items-center">
            <button
              onClick={toggleRoster}
              className="pixel-button !py-1 !px-2 text-sm"
              aria-label="Toggle character roster"
            >
              Change Character
            </button>
          </div>
        </div>
      </div>

      {/* Character roster */}
      <div
        className={`bg-chilis bg-opacity-90 transition-all duration-300 overflow-hidden fixed left-0 w-full z-40 ${
          showRoster ? 'max-h-96' : 'max-h-0'
        }`}
        style={{ top: '72px' }}
      >
        <div className="w-full bg-chilis bg-opacity-90 p-4 pixel-border shadow-md">
          <h2 className="font-pixel text-white text-xl mb-4">Choose Your Character:</h2>
          <div className="flex flex-wrap gap-4 justify-center pb-10">
            {characters.map((character) => (
              <CharacterSprite
                key={character}
                character={character}
                isActive={character === currentCharacter}
                onClick={() => handleCharacterSelect(character)}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Spacer to prevent overlap */}
      <div style={{ height: showRoster ? '215px' : '72px' }} />

      {/* Dundies points counter */}
      <div className="bg-pretzel p-2 text-right">
        <div className="flex items-center justify-end font-pixel text-white">
          <img
            src="https://i.ibb.co/LCfJ8bK/dundie-trophy.png"
            alt="Dundie Trophy"
            className="w-6 h-6 mr-1"
          />
          <span>Dundies: {dundiesPoints}</span>
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex-1 overflow-y-auto p-4 bg-officeBackground bg-paper-texture">
        <div className="absolute inset-0 flex items-center justify-center opacity-5 pointer-events-none">
          <div className="text-9xl font-bold tracking-widest rotate-12">
            DUNDER MIFFLIN
          </div>
        </div>

        <div className="max-w-3xl mx-auto">
          {messages.map((message) => (
            <ChatBubble key={message.id} message={message} />
          ))}
          {loading && (
            <div className="flex justify-start my-2">
              <div className="bg-schrute bg-opacity-70 text-white p-3 pixel-border rounded font-pixel">
                <p className="typing-indicator">{currentCharacter} is typing</p>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Chat input */}
      <div className="bg-pretzel p-4 pixel-border">
        <div className="max-w-3xl mx-auto flex items-center">
          <div className="hidden sm:block mr-3">
            <div className="w-12 h-12 pixel-border overflow-hidden bg-black">
              <img
                src={`https://i.ibb.co/${
                  currentCharacter === 'Michael'
                    ? 'YQKWTkC/michael'
                    : currentCharacter === 'Jim'
                    ? 'cY25vBN/jim'
                    : currentCharacter === 'Pam'
                    ? 'ZH1tygH/pam'
                    : currentCharacter === 'Dwight'
                    ? 'kS5Z72v/dwight'
                    : currentCharacter === 'Angela'
                    ? 'VSxFvf6/angela'
                    : currentCharacter === 'Oscar'
                    ? 'NpDjyC1/oscar'
                    : currentCharacter === 'Kevin'
                    ? 'PGM9QnW/kevin'
                    : currentCharacter === 'Stanley'
                    ? 'G5T4H3W/stanley'
                    : currentCharacter === 'Creed'
                    ? 'DbKH3vY/creed'
                    : 'dkKL9Vf/toby'
                }.png`}
                alt={currentCharacter}
                className="w-full h-full object-cover"
              />
            </div>
          </div>

          <form onSubmit={handleSubmit} className="flex-1 flex">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={`Message ${currentCharacter}...`}
              className="flex-1 pixel-input mr-2"
              aria-label="Message input"
            />
            <button
              type="submit"
              className="pixel-button bg-schrute text-white !py-2 !px-4"
              disabled={loading}
              aria-label="Send message"
              onClick={() => {
                const sendSound = new Audio(
                  'https://assets.mixkit.co/sfx/preview/mixkit-message-pop-alert-2354.mp3'
                );
                sendSound.volume = 0.2;
                sendSound.play();
              }}
            >
              Send
            </button>
          </form>
        </div>

        <div className="text-xs text-center mt-2 text-black font-pixel">
          Tip: Type /switch [Character] to change characters
        </div>
      </div>

      {achievements.map((achievement) => (
        <AchievementPopup
          key={achievement.id}
          achievement={achievement}
          onComplete={() => removeAchievement(achievement.id)}
        />
      ))}
    </div>
  );
};

export default Chat;