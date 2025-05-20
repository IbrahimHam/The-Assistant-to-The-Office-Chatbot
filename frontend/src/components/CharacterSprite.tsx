
import { Character } from '@/types';
import { cn } from '@/lib/utils';

interface CharacterSpriteProps {
  character: Character;
  isActive: boolean;
  onClick: () => void;
}

const CharacterSprite = ({ character, isActive, onClick }: CharacterSpriteProps) => {
  const getCharacterImage = (character: Character): string => {
    // Pixel art character placeholders
    const characterImages: Record<Character, string> = {
      Michael: "https://i.postimg.cc/06pqCKCM/micheal.png",
      Jim: "https://i.postimg.cc/hJFBx1VR/Jim.png",
      Pam: "https://i.postimg.cc/gL0qG5cV/pam.png",
      Dwight: "https://i.postimg.cc/CB7sTY0M/dwight.png",
      Angela: "https://i.postimg.cc/d1vZZbj1/angela.png",
      Oscar: "https://i.postimg.cc/njpGPr6w/oscar.png",
      Kevin: "https://i.postimg.cc/nVGY2kXg/kevin.png",
      Stanley: "https://i.postimg.cc/QVNqgBgF/stanley.png",
      Creed: "https://i.postimg.cc/SX0rcDV4/creed.png",
      Toby: "https://i.postimg.cc/w3WQw6F0/toby.png"
    };
    
    return characterImages[character];
  };

  const getCharacterQuote = (character: Character): string => {
    const quotes: Record<Character, string> = {
      Michael: "I'm not superstitious, but I am a little stitious.",
      Jim: "Bears. Beets. Battlestar Galactica.",
      Pam: "There's a lot of beauty in ordinary things.",
      Dwight: "Identity theft is not a joke, Jim!",
      Angela: "I don't have a headache. I'm just preparing.",
      Oscar: "Actually...",
      Kevin: "Why waste time say lot word when few word do trick?",
      Stanley: "Did I stutter?",
      Creed: "I've been involved in a number of cults.",
      Toby: "Why you gotta be so mean to me?"
    };
    
    return quotes[character];
  };

  const handleClick = () => {
    // Play click sound
    const clickSound = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-video-game-retro-click-237.mp3');
    clickSound.volume = 0.3;
    clickSound.play();
    onClick();
  };

  return (
    <div className="relative group">
      <div
        className={cn(
          "character-sprite pixel-border overflow-hidden",
          isActive && "ring-4 ring-chilis animate-pulse-pixel"
        )}
        onClick={handleClick}
        title={character}
      >
        <img 
          src={getCharacterImage(character)} 
          alt={character} 
          className="w-full h-full object-cover"
        />
      </div>
      
      <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute -bottom-16 left-1/2 transform -translate-x-1/2 bg-black text-white p-2 rounded w-32 text-xs z-10">
        <p className="font-pixel text-center">
          <span className="text-dundies">{character}:</span> "{getCharacterQuote(character)}"
        </p>
      </div>
    </div>
  );
};

export default CharacterSprite;
