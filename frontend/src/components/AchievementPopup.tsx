
import { Achievement } from '@/types';
import { useEffect, useState } from 'react';

interface AchievementPopupProps {
  achievement: Achievement;
  onComplete: () => void;
}

const AchievementPopup = ({ achievement, onComplete }: AchievementPopupProps) => {
  const [visible, setVisible] = useState(true);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(false);
      setTimeout(onComplete, 300); // Allow time for exit animation
    }, 3000);
    
    return () => clearTimeout(timer);
  }, [onComplete]);
  
  return (
    <div className={`achievement ${visible ? 'opacity-100' : 'opacity-0'} transition-opacity duration-300`}>
      <div className="flex items-center">
        <span className="text-2xl mr-2">{achievement.icon}</span>
        <div>
          <h3 className="text-lg font-bold">{achievement.title}</h3>
          <p className="text-sm">{achievement.description}</p>
        </div>
      </div>
    </div>
  );
};

export default AchievementPopup;
