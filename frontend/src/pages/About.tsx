
import { Link } from 'react-router-dom';
import { useEffect } from 'react';

const About = () => {
  useEffect(() => {
    // Play paper sound on mount
    const paperSound = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-paper-slide-1530.mp3');
    paperSound.volume = 0.4;
    paperSound.play();
  }, []);

  return (
    <div className="min-h-screen bg-officeBackground bg-paper-texture flex flex-col items-center p-4 sm:p-8">
      <div className="max-w-3xl w-full">
        {/* Header */}
        <div className="mb-8 text-center">
          <Link 
            to="/" 
            className="pixel-button inline-block mb-6"
            onClick={() => {
              const clickSound = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-video-game-retro-click-237.mp3');
              clickSound.volume = 0.3;
              clickSound.play();
            }}
          >
            Back to Home
          </Link>
          
          <div className="pixel-border bg-white p-6 relative">
            {/* Memo header */}
            <div className="border-b-4 border-black pb-4 mb-4">
              <img 
                src="https://i.ibb.co/g7Rncnz/dunder-mifflin-logo.png" 
                alt="Dunder Mifflin" 
                className="h-16 mx-auto mb-2" 
              />
              <h1 className="font-pixel text-3xl text-dundies">Dunder Mifflin Memo</h1>
              <p className="font-pixel text-sm text-black">ATTENTION: All Branches</p>
            </div>
            
            {/* Content */}
            <div className="text-left">
              <h2 className="font-pixel text-2xl mb-4 text-chilis">Project Description</h2>
              <p className="mb-4 font-pixel text-black">
              The Assistant to The Office Chatbot lets you chat with your favorite Dunder Mifflin characters
               — in character, emotionally aware, and hilariously awkward.
              </p>
              
              <p className="mb-6 font-pixel text-black">
              Powered by Groq’s LLaMA 3 and real show dialogue, the chatbot retrieves actual scenes,
              understands your emotional tone and sarcasm,
              and responds just like Pam, Dwight, Michael, and others.
              </p>

              <p className="mb-6 font-pixel text-black">
              Whether you're chatting in the console or through our pixel-perfect frontend,
               you'll relive Scranton’s finest chaos — one awkward moment at a time.
              </p>
              
              <h2 className="font-pixel text-2xl mb-4 text-chilis">Meet the Team</h2>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-6">
                {[
                  {
                    name: "Ibrahim Hammoud", 
                    role: "NLP Specialist", 
                    image: "https://i.ibb.co/YQKWTkC/michael.png",
                    link: "https://linkedin.com/in/johndoe"
                  },
                  {
                    name: "Nikita Buss", 
                    role: "Character AI Developer", 
                    image: "https://i.ibb.co/ZH1tygH/pam.png",
                    link: "https://linkedin.com/in/janesmith"
                  },
                  {
                    name: "Daniel G", 
                    role: "Dialogue Systems Engineer", 
                    image: "https://i.ibb.co/cY25vBN/jim.png",
                    link: "https://linkedin.com/in/alexlee"
                  }
                ].map((teammate, index) => (
                  <div key={index} className="text-center">
                    <div className="w-20 h-20 mx-auto mb-2 pixel-border overflow-hidden">
                      <img 
                        src={teammate.image} 
                        alt={teammate.name} 
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <h3 className="font-pixel text-xl text-schrute">{teammate.name}</h3>
                    <p className="font-pixel text-sm text-black">{teammate.role}</p>
                    <a 
                      href={teammate.link} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-xs font-pixel text-dundies hover:underline"
                      onClick={(e) => {
                        e.preventDefault(); // Prevent actual navigation since this is a demo
                        const linkSound = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-message-pop-alert-2354.mp3');
                        linkSound.volume = 0.3;
                        linkSound.play();
                      }}
                    >
                      LinkedIn Profile
                    </a>
                  </div>
                ))}
              </div>
              
              <div className="text-center mt-8">
                <a 
                  href="https://github.com/IbrahimHam/The-Assistant-to-The-Office-Chatbot" 
                  className="pixel-button bg-dundies inline-flex items-center"
                  onClick={(e) => {
                    // e.preventDefault(); // Prevent actual navigation since this is a demo
                    const githubSound = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-arcade-game-jump-coin-216.mp3');
                    githubSound.volume = 0.3;
                    githubSound.play();
                  }}
                >
                  <span className="mr-2">💾</span> View on GitHub
                </a>
              </div>
              
              {/* "Signed" by Michael */}
              <div className="mt-8 text-right font-marker text-xl text-black">
                <p>Michael Scott</p>
                <p className="text-sm">Regional Manager</p>
              </div>
            </div>
            
            {/* Stapler in Jello easter egg */}
            <div className="absolute -bottom-6 -right-6 w-16 h-16">
              <img 
                src="https://i.ibb.co/k0jYrGS/stapler-jello.png" 
                alt="Stapler in Jello" 
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
