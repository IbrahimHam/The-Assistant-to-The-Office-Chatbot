
import { Link } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { Button } from "@/components/ui/button";

const Index = () => {
  const [loaded, setLoaded] = useState(false);
  
  useEffect(() => {
    // Load animation once component mounts
    setLoaded(true);
    
    // Play intro sound
    const introSound = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-game-level-completed-2059.mp3');
    introSound.volume = 0.4;
    introSound.play();
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center py-10 px-4 bg-officeBackground relative overflow-hidden">
      {/* Pixelated background overlay */}
      <div className="absolute inset-0 opacity-10 bg-pixel-grid pointer-events-none"></div>
      
      {/* Paper texture and Dunder Mifflin watermark */}
      <div className="absolute inset-0 bg-paper-texture opacity-30 pointer-events-none"></div>
      <div className="absolute inset-0 flex items-center justify-center opacity-5 pointer-events-none">
        <div className="text-9xl font-bold tracking-widest rotate-12">
          DUNDER MIFFLIN
        </div>
      </div>
      
      {/* Main content */}
      <div className={`z-10 text-center transition-transform duration-500 ${loaded ? 'translate-y-0' : 'translate-y-16'}`}>
        {/* World's Best Boss mug decoration */}
        <div className="w-28 h-28 mx-auto mb-4 relative">
          <img 
            src="https://i.ibb.co/w7QW1yk/worlds-best-boss-mug.png" 
            alt="World's Best Boss Mug" 
            className="w-full h-full object-contain"
          />
        </div>
        
        <h1 className="text-4xl sm:text-6xl font-pixel text-chilis mb-4">
          The Assistant to <br className="md:hidden" />
          <span className="text-dundies">The Office</span> Chatbot
        </h1>
        
        <h2 className="text-xl sm:text-2xl font-pixelify mb-8 text-black">
          Join Scranton's Finest for an AI-Powered Adventure!
        </h2>
        
        {/* Pulsing button */}
        <div className="flex flex-col sm:flex-row justify-center gap-4 mb-8">
          <Link to="/chat">
            <button 
              className="pixel-button bg-dundies animate-pulse-pixel"
              onClick={() => {
                const buttonSound = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-arcade-game-jump-coin-216.mp3');
                buttonSound.volume = 0.3;
                buttonSound.play();
              }}
            >
              Start Chatting!
            </button>
          </Link>
          
          <Link to="/about">
            <button 
              className="pixel-button bg-schrute"
              onClick={() => {
                const buttonSound = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-arcade-game-jump-coin-216.mp3');
                buttonSound.volume = 0.3;
                buttonSound.play();
              }}
            >
              About
            </button>
          </Link>
        </div>
        
        {/* Character preview */}
        <div className="mt-8">
          <p className="font-pixel text-xl animate-blink text-black">Press Start to Chat with The Office!</p>
          
          <div className="flex flex-wrap justify-center gap-3 mt-4">
            {["Michael", "Jim", "Pam", "Dwight", "Angela"].map((char) => (
              <div key={char} className="w-12 h-12 pixel-border overflow-hidden">
                <img 
                  src={`https://i.ibb.co/${
                    char === "Michael" ? "YQKWTkC/michael" :
                    char === "Jim" ? "cY25vBN/jim" :
                    char === "Pam" ? "ZH1tygH/pam" :
                    char === "Dwight" ? "kS5Z72v/dwight" :
                    "VSxFvf6/angela"
                  }.png`} 
                  alt={char} 
                  className="w-full h-full object-cover"
                />
              </div>
            ))}
          </div>
        </div>
      </div>
      
      <footer className="absolute bottom-4 text-center font-pixel text-sm text-black">
        <p>© 2025 Dunder Mifflin Paper Company, Inc.</p>
      </footer>
    </div>
  );
};

export default Index;
